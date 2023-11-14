#!/usr/bin/env python

import logging
import os
import traceback

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transcript_file_format import to_json as transcript_to_json

from cdp_transcription import __version__
from cdp_transcription.transcribe import TranscriptionConfig, WhisperModel

###############################################################################

log = logging.getLogger(__name__)

###############################################################################
# Basic App Setup

app = FastAPI()

_PRELOADED_NLP = None
_PRELOADED_MODEL = None

###############################################################################
# App routes


# Status handling
class StatusResponse(BaseModel):
    status: str
    version: str


@app.get("/status/")
async def root() -> StatusResponse:
    """Call with http://localhost:{port}/status/."""
    return StatusResponse(
        status="alive",
        version=__version__,
    )


# Transcription handling
class TranscriptionSuccess(BaseModel):
    transcript_storage_uri: str


class TranscriptionError(BaseModel):
    error: str
    traceback: str


DEFAULT_TRANSCRIPTION_CONFIG = TranscriptionConfig()


@app.post("/transcribe")
async def transcribe(
    audio_uri: str,
    output_filepath: str,
    config: TranscriptionConfig = DEFAULT_TRANSCRIPTION_CONFIG,
) -> TranscriptionSuccess | TranscriptionError:
    """Call with http://localhost:{port}/transcribe."""
    try:
        transcription_handler = WhisperModel(
            config=config,
            nlp=_PRELOADED_NLP,
            model=_PRELOADED_MODEL,
        )

        transcript = transcription_handler.transcribe(
            audio_uri=audio_uri,
        )

        # Store to file
        transcript_to_json(
            transcript=transcript,
            output_filepath=output_filepath,
        )

        return TranscriptionSuccess(
            transcript_storage_uri=output_filepath,
        )

    except Exception as e:
        return TranscriptionError(
            error=str(e),
            traceback=traceback.format_exc(),
        )


###############################################################################
# Bin handler


def main() -> None:
    global _PRELOADED_NLP
    global _PRELOADED_MODEL

    port = os.environ.get("PORT", 8080)

    # Setup logging
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    lvl = getattr(logging, log_level.upper())
    logging.basicConfig(
        level=lvl,
        format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
    )

    # Preload models
    preload_nlp = os.environ.get("PRELOAD_NLP", True)
    preload_model = os.environ.get("PRELOAD_MODEL", True)
    preload_model_config = os.environ.get("PRELOAD_MODEL_CONFIG", None)

    if preload_nlp:
        _PRELOADED_NLP = WhisperModel._load_spacy_model()
    if preload_model:
        _PRELOADED_MODEL = WhisperModel(config=DEFAULT_TRANSCRIPTION_CONFIG)
        if preload_model_config:
            raise NotImplementedError(
                "Preloading whisper model with custom config is not yet supported."
            )

    # Run server
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
