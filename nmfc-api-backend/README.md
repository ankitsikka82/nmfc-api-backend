# NMFC FastAPI Backend

## Setup

1. Copy `.env.template` to `.env` and fill in your OpenAI API key.
2. Run locally:
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

3. Deploy to Railway:
- Connect GitHub
- Set `OPENAI_API_KEY` in environment variables
