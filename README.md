# StyleMate: AI Dress Code Agent

An AI agent that suggests outfits for a boy, a girl, or both, from a text description and an optional photo. It uses the person's own reasons, the occasion, budget and weather, and then shows shopping links and similar outfit photos for each suggested item.

## Features
- Outfit suggestions with or without a photo
- Takes the person's own reasons and preferences into account
- Shopping links to Amazon, Flipkart, Myntra, Ajio, Meesho and Google Shopping
- Similar outfit photos from Pexels, credited to their photographers
- Retries and optional fallback model when the AI service is busy

## Tech
Python, Flask, Google Gemini API, Pexels API

## Setup
1. `python -m venv .venv` and activate it (`source .venv/bin/activate` on Mac/Linux)
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your Gemini and Pexels keys
4. `python app.py`, then open http://127.0.0.1:5001

## Notes
- Photos are kept in memory only and never saved.
- The agent uses only visible style details and never guesses age, health or religion.
- Prices are rough estimates. Tap a store link for live prices.
