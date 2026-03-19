# ai-tel

`ai-tel` is now organized as a small Python project instead of a single loose script.
It provides five core features:

- language detection
- basic text analysis
- template-based text generation
- microphone recording plus OpenAI `gpt-4o-transcribe`
- text-to-speech playback with OpenAI `gpt-4o-mini-tts`

## Project structure

```text
ai-tel/
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|-- generated_wav/
|-- pyproject.toml
|-- README.md
|-- requirements.txt
|-- ai_text_generator.py
|-- scripts/
|   `-- dev.ps1
|-- src/
|   `-- ai_tel/
|       |-- __init__.py
|       |-- cli.py
|       |-- gui.py
|       |-- openai_speech.py
|       |-- openai_tts.py
|       |-- processor.py
|       `-- tts_gui.py
`-- tests/
    |-- test_cli.py
    |-- test_openai_speech.py
    |-- test_openai_tts.py
    `-- test_processor.py
```

## Install

```bash
python -m pip install -r requirements.txt
```

## Configure

The project reads `OPENAI_API_KEY` from `.env` automatically.

## Speech To Text GUI

Start the local speech-to-text test UI:

```bash
python -m ai_tel.gui
```

or:

```bash
ai-tel-gui
```

How it works:

- choose the language hint from the dropdown list
- click `Start Recording`
- speak into the default microphone
- click `Stop Recording`
- the app uploads the recorded WAV audio to OpenAI `gpt-4o-transcribe`
- the transcript is shown in the window
- click `Go To Text To Speech` to switch to the TTS window

## Text To Speech GUI

Start the local TTS test UI:

```bash
python -m ai_tel.tts_gui
```

or:

```bash
ai-tel-tts-gui
```

How it works:

- enter text in the input box
- choose gender and age style from the dropdowns
- click `Generate And Play`
- the app generates speech with OpenAI `gpt-4o-mini-tts`
- the generated WAV file is saved into `generated_wav/`
- the saved WAV file is then played locally
- click `Go To Speech To Text` to switch back to the transcription window

Note:

- OpenAI's TTS API provides built-in voices and instruction-based style control, not direct age/gender parameters.
- This app maps your selected gender and age to a voice profile and speaking instructions. That mapping is application-level inference, not an official OpenAI age/gender setting.

## CLI

```bash
python ai_text_generator.py listen --timeout 8 --culture ja-JP --process transcript
python ai_text_generator.py listen --timeout 8 --culture en-US --process analyze
python ai_text_generator.py listen --timeout 8 --culture zh-CN --process generate --language japanese --length 100
```

## Test

```bash
python -m pytest
```
