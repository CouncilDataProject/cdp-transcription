#!/usr/bin/env python

import logging
import os
import traceback

import uvicorn
from fastapi import FastAPI
from faster_whisper import WhisperModel as FasterWhisper
from pydantic import BaseModel
from transcript_file_format import to_json as transcript_to_json

from cdp_transcription import __version__
from cdp_transcription.transcribe import TranscriptionConfig, WhisperModel

###############################################################################

log = logging.getLogger(__name__)

###############################################################################
# Basic App Setup

app = FastAPI()

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
        # Handle cached model or load new
        if _PRELOADED_MODEL:
            transcription_handler = _PRELOADED_MODEL
        else:
            transcription_handler = WhisperModel(config=config)

        # Transcribe
        transcript = transcription_handler.transcribe(
            audio_uri=audio_uri,
        )

        # Store to file
        transcript_to_json(
            transcript=transcript,
            path=output_filepath,
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
    global _PRELOADED_MODEL

    # Setup logging
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    lvl = getattr(logging, log_level.upper())
    logging.basicConfig(
        level=lvl,
        format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
    )

    # Preload models
    preload_model = os.environ.get("PRELOAD_MODEL", True)
    if preload_model:
        nlp = WhisperModel._try_load_spacy_model()
        model = FasterWhisper(
            model_size_or_path=DEFAULT_TRANSCRIPTION_CONFIG.model,
            device=DEFAULT_TRANSCRIPTION_CONFIG.device,
            compute_type=DEFAULT_TRANSCRIPTION_CONFIG.compute_type,
        )

    _PRELOADED_MODEL = WhisperModel(
        config=DEFAULT_TRANSCRIPTION_CONFIG,
        model=model,
        nlp=nlp,
    )

    # Run server
    port = os.environ.get("PORT", 8080)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
