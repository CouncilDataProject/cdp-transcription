# CDP Whisper Server

A fastapi docker whisper server that returns transcripts using [transcript-file-format](https://github.com/CouncilDataProject/transcript-file-format).

## Installation

```bash
pip install .
conda install -c anaconda cudnn -y
```

## Usage

```bash
run-cdp-transcription-server
```

### Documentation

Once the server is running, the API documentation can be found at [http://localhost:8080/docs](http://localhost:8080/docs).

### Reading Produced Transcripts

Transcripts are stored in JSON but have a defined Python model if you want to enforce typing. The underlying model used is the [transcript-file-format](https://github.com/councildataproject/transcript-file-format).

To read in produced transcripts using this model:

```python
from transcript_file_format import from_json

transcript = from_json("example.json")

# transcript details
# Transcript(sentences=[...] (n=633), generator='CDP Transcription -- version 0.1.dev0+d20231114 -- Whisper Model Name 'small'', confidence=None, session_datetime=None, created_datetime='2023-11-14T17:35:59.425462', annotations=None)
```

## TODO

TODO: update to [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper)