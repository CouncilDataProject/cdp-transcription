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

## TODO

TODO: update to [insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper)