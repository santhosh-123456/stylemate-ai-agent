# StyleMate: AI Dress Code Agent

Suggests outfits for a boy, a girl, or both, with or without a photo, using the person's own reasons and preferences. Built with Gemini (Google AI Studio), Python and Flask.

## Run
1. Get an API key: https://aistudio.google.com/apikey
2. `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your key (never commit `.env`).
4. `python app.py`, then open http://127.0.0.1:5001

## Files
- `system_prompt.md`: the agent's behavior and rules
- `app.py`: web app that sends text and an optional photo to Gemini
