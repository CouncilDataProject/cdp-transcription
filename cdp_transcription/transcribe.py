#!/usr/bin/env python

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import spacy
from faster_whisper import WhisperModel as FasterWhisper
from pydantic import BaseModel
from pydub import AudioSegment
from spacy.cli.download import download as download_spacy_model
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
from tqdm import tqdm
from transcript_file_format import Sentence, Transcript, Word

from . import __version__

if TYPE_CHECKING:
    from spacy.language import Language

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

spacy.prefer_gpu()

###############################################################################


class TranscriptionConfig(BaseModel):
    model: str = "small"
    language: str = "en"
    compute_type: str = "float32"
    device: str = "auto"


class WhisperModel:
    @staticmethod
    def _load_spacy_model() -> Language:
        nlp = spacy.load(
            "en_core_web_trf",
            # Only keep the parser
            # We are only using this for sentence parsing
            disable=[
                "tagger",
                "ner",
                "lemmatizer",
                "textcat",
            ],
        )

        # Do not split hyphenated words and numbers
        # "re-upping" should not be split into ["re", "-", "upping"].
        # Credit: https://stackoverflow.com/a/59996153
        def custom_tokenizer(nlp: Language) -> Tokenizer:
            inf = list(nlp.Defaults.infixes)
            inf.remove(r"(?<=[0-9])[+\-\*^](?=[0-9-])")
            infixes = (*inf, r"(?<=[0-9])[+*^](?=[0-9-])", r"(?<=[0-9])-(?=-)")
            infixes = tuple(
                [x for x in infixes if "-|–|—|--|---|——|~" not in x]  # noqa: RUF001
            )
            infix_re = compile_infix_regex(infixes)

            return Tokenizer(
                nlp.vocab,
                prefix_search=nlp.tokenizer.prefix_search,
                suffix_search=nlp.tokenizer.suffix_search,
                infix_finditer=infix_re.finditer,
                token_match=nlp.tokenizer.token_match,
                rules=nlp.Defaults.tokenizer_exceptions,
            )

        nlp.tokenizer = custom_tokenizer(nlp)
        return nlp

    @staticmethod
    def _try_load_spacy_model() -> Language | None:
        try:
            return WhisperModel._load_spacy_model()
        except Exception:
            download_spacy_model("en_core_web_trf")
            return WhisperModel._load_spacy_model()

    @staticmethod
    def _clean_word(word: str) -> str:
        return re.sub(r"[^\w\/\-\']+", "", word).lower()

    def __init__(
        self,
        config: TranscriptionConfig,
        model: FasterWhisper | None = None,
        nlp: Language | None = None,
    ):
        """
        Initialize an OpenAI Whisper Model Transcription processor.

        Parameters
        ----------
        config: TranscriptionConfig
            The configuration for transcription.
        model: FasterWhisper | None
            Optional pre-loaded whisper model.
        nlp: Language | None
            Optional pre-loaded spacy model.
        """
        # Handle large -> large v2
        if config.model == "large":
            config.model = "large-v3"

        # Set config
        self.config = config

        # Load whisper model
        if model is None:
            log.info(f"Loading whisper model '{config.model}'")
            self.model = FasterWhisper(
                model_size_or_path=config.model,
                device=config.device,
                compute_type=config.compute_type,
            )
        else:
            self.model = model

        # Init spacy
        if nlp is not None:
            self.nlp = nlp
        else:
            self.nlp = self._try_load_spacy_model()

    def transcribe(
        self,
        audio_uri: str | Path,
    ) -> Transcript:
        """
        Transcribe audio from file and return a Transcript model.

        Parameters
        ----------
        audio_uri: Union[str, Path]
            The uri to the audio file or caption file to transcribe.

        Returns
        -------
        Transcript
            The transcript of the audio file.
        """
        log.info(f"Transcribing '{audio_uri}'")
        segments, _ = self.model.transcribe(
            audio_uri,
            language=self.config.language,
            word_timestamps=True,
        )
        timestamped_words_with_meta = []
        for segment in tqdm(segments, desc="Transcribing segment..."):
            for word in segment.words:
                word_text = word.word
                word_text = word_text.replace("♪", "")
                word_text = word_text.replace("≫", "")
                word_text = re.sub(r" +", " ", word_text)
                word_text = re.sub(r"( +)(\.)", ".", word_text)
                word_text = word_text.strip()
                if len(word_text) > 0:
                    timestamped_words_with_meta.append(
                        {
                            "text": word_text,
                            "start": word.start,
                            "end": word.end,
                        }
                    )

        # For some reason, whisper sometimes returns segments with
        # start and end times that are impossible
        # i.e. start and end of 185 second when the total audio duration is 180 seconds
        # Fix all timestamps by rescaling to audio duration
        # This is a hack -- but all of the word level timestamps are a hack anyway...
        whisper_reported_duration = timestamped_words_with_meta[-1]["end"]
        file_reported_duration = AudioSegment.from_file(audio_uri).duration_seconds

        # Scale to between 0 and 1
        # Then rescale to real duration
        log.info("Ensuring timestamps fit within audio")
        for word_with_meta in timestamped_words_with_meta:
            # Scale to between 0 and 1
            word_with_meta["start"] = (
                word_with_meta["start"] / whisper_reported_duration
            )
            word_with_meta["end"] = word_with_meta["end"] / whisper_reported_duration

            # Rescale to real duration
            word_with_meta["start"] = word_with_meta["start"] * file_reported_duration
            word_with_meta["end"] = word_with_meta["end"] * file_reported_duration

        # Process all text
        joined_all_words = " ".join(
            [word_with_meta["text"] for word_with_meta in timestamped_words_with_meta]
        )
        joined_all_words = re.sub(r" +", " ", joined_all_words).strip()
        doc = self.nlp(joined_all_words)

        # Process sentences
        sentences_with_word_metas = []
        current_word_index_start = 0
        log.info("Constructing sentences with word metadata")
        for doc_sent in doc.sents:
            doc_sent_text = doc_sent.text.strip()
            # Sometimes spacy produces a doc sentence that is just a period or comma.
            # This sentence is attached to the end of the word
            # in the timestamped words with metas list
            # We can simply ignore those odd sentences
            if any(c == doc_sent_text for c in [".", ","]):
                continue

            log.debug(f"Doc sent: '{doc_sent_text}'")
            # Split the sentence
            doc_sent_words = doc_sent_text.split(" ")

            # Find the words
            word_subset = timestamped_words_with_meta[
                current_word_index_start : current_word_index_start
                + len(doc_sent_words)
            ]
            log.debug(f"\tWords: {[w_w_m['text'] for w_w_m in word_subset]}")

            # Append the words
            sentences_with_word_metas.append(word_subset)

            # Increase the current word index start
            current_word_index_start = current_word_index_start + len(doc_sent_words)

        # Remove any length zero sentences
        sentences_with_word_metas = [
            sentence_with_word_metas
            for sentence_with_word_metas in sentences_with_word_metas
            if len(sentence_with_word_metas) > 0
        ]

        # Reformat data to our structure
        structured_sentences: list[Sentence] = []
        log.info("Converting sentences with word meta to transcript format")
        for sentence_with_word_metas in sentences_with_word_metas:
            # Join all the sentence text
            sentence_text = " ".join(
                [word_with_meta["text"] for word_with_meta in sentence_with_word_metas]
            ).strip()

            # Make sure the first letter is capitalized
            # NOTE: we cannot use the `capitalize` string function
            # because it will lowercase the rest of the text
            sentence_text = sentence_text[0].upper() + sentence_text[1:]

            # Create the sentence object
            structured_sentences.append(
                Sentence(
                    start_time=sentence_with_word_metas[0]["start"],
                    end_time=sentence_with_word_metas[-1]["end"],
                    text=sentence_text,
                    words=[
                        Word(
                            start_time=word_with_meta["start"],
                            end_time=word_with_meta["end"],
                            text=self._clean_word(word_with_meta["text"]),
                        )
                        for word_with_meta in sentence_with_word_metas
                    ],
                )
            )

        # Return complete transcript object
        return Transcript(
            generator=(
                f"CDP Transcription "
                f"-- version {__version__} "
                f"-- Whisper Model Name '{self.config.model}'"
            ),
            created_datetime=datetime.utcnow().isoformat(),
            sentences=structured_sentences,
        )
