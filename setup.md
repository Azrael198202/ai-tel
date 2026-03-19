# Setup

## 1. Create a virtual environment

```bash
python -m venv .venv
```

## 2. Activate it

Windows:

```bash
.venv\Scripts\activate
```

## 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## 4. Make sure `.env` contains your OpenAI key

```text
OPENAI_API_KEY=your_api_key_here
```

## 5. Start the GUI test app

```bash
python -m ai_tel.gui
```

## 6. Or use the CLI

```bash
python ai_text_generator.py listen --timeout 8 --culture ja-JP --process transcript
```