"""StyleMate: AI dress code agent (Flask + Gemini). Run with: python app.py"""
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from google import genai
from google.genai import types

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "").strip()
RETRIES = 4  # attempts per model when Google says "busy"


def is_busy_error(e):
    """True for temporary Google-side problems (503 overloaded, 500, 429 rate limit)."""
    code = getattr(e, "code", None)
    return code in (429, 500, 502, 503, 504) or "UNAVAILABLE" in str(e)
SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text(encoding="utf-8")
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024  # 6 MB upload limit

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StyleMate</title>
<style>
  body{font-family:-apple-system,Segoe UI,sans-serif;max-width:640px;margin:32px auto;padding:0 16px;color:#222}
  h1{margin-bottom:4px} p.sub{color:#666;margin-top:0}
  label{display:block;margin:14px 0 4px;font-weight:600}
  input,select,textarea{width:100%;padding:9px;border:1px solid #ccc;border-radius:8px;font:inherit;box-sizing:border-box}
  button{margin-top:18px;padding:11px 20px;border:0;border-radius:8px;background:#5b3df5;color:#fff;font:inherit;cursor:pointer}
  button:disabled{opacity:.6}
  #out{margin-top:24px;padding:16px;background:#f6f5ff;border-radius:10px;display:none}
  .err{color:#b00020}
</style></head><body>
<h1>StyleMate</h1><p class="sub">Get an outfit suggestion, with or without a photo.</p>
<form id="f">
  <label>Who is it for?</label>
  <select name="who"><option>Boy</option><option>Girl</option><option>Both</option><option selected>Not specified</option></select>
  <label>Occasion</label><input name="occasion" placeholder="Job interview, wedding, college farewell...">
  <label>About you (optional)</label><textarea name="description" rows="2" placeholder="Height, build, skin tone, colors you like..."></textarea>
  <label>Your reasons / preferences (optional)</label><textarea name="reasons" rows="2" placeholder="I want something comfortable because..."></textarea>
  <label>Budget (optional)</label><input name="budget" placeholder="e.g. Rs 2000">
  <label>Weather / season (optional)</label><input name="weather" placeholder="e.g. hot and humid">
  <label>Photo (optional)</label><input type="file" name="photo" accept="image/jpeg,image/png,image/webp">
  <button id="b" type="submit">Suggest outfit</button>
</form>
<div id="out"></div>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
const f=document.getElementById('f'),out=document.getElementById('out'),b=document.getElementById('b');
f.addEventListener('submit',async e=>{
  e.preventDefault(); b.disabled=true; b.textContent='Thinking...';
  out.style.display='block'; out.textContent='Working on your outfit...';
  try{
    const r=await fetch('/suggest',{method:'POST',body:new FormData(f)});
    const d=await r.json();
    if(d.error){out.innerHTML='<span class="err"></span>';out.firstChild.textContent=d.error;}
    else{out.innerHTML=(window.marked?marked.parse(d.result):d.result.replace(/\\n/g,'<br>'));}
  }catch(err){out.innerHTML='<span class="err">Could not reach the server.</span>';}
  b.disabled=false; b.textContent='Suggest outfit';
});
</script></body></html>"""


@app.route("/")
def index():
    return HTML


@app.route("/suggest", methods=["POST"])
def suggest():
    f = request.form
    text = (
        "Suggest a dress code.\n"
        f"For: {f.get('who', 'Not specified')}\n"
        f"Occasion: {f.get('occasion', '').strip() or 'not given'}\n"
        f"About the person: {f.get('description', '').strip()[:500] or 'none'}\n"
        f"Their reasons/preferences: {f.get('reasons', '').strip()[:500] or 'none'}\n"
        f"Budget: {f.get('budget', '').strip()[:100] or 'not given'}\n"
        f"Weather/season: {f.get('weather', '').strip()[:100] or 'not given'}"
    )
    parts = [text]

    photo = request.files.get("photo")
    if photo and photo.filename:
        if photo.mimetype not in IMAGE_TYPES:
            return jsonify(error="Photo must be a JPG, PNG or WEBP image."), 400
        parts.append(types.Part.from_bytes(data=photo.read(), mime_type=photo.mimetype))
        parts.append("A photo of the person is attached.")  # kept in memory only, never saved

    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.7)
    # Try the main model, then the optional backup model from .env (GEMINI_FALLBACK_MODEL).
    models = [MODEL] + ([FALLBACK_MODEL] if FALLBACK_MODEL and FALLBACK_MODEL != MODEL else [])
    last_error = None
    for model in models:
        for attempt in range(RETRIES):
            try:
                resp = client.models.generate_content(model=model, contents=parts, config=config)
                return jsonify(result=resp.text or "No answer came back. Please try again.")
            except Exception as e:
                last_error = e
                app.logger.warning("Gemini %s attempt %d failed: %s", model, attempt + 1, e)
                if not is_busy_error(e):
                    return jsonify(error=f"AI error: {e}"), 502  # real problem: retrying won't help
                if attempt < RETRIES - 1:
                    time.sleep(2 ** (attempt + 1))  # wait 2s, 4s, 8s...

    return jsonify(error="Google's AI model is very busy right now. "
                         "Please wait a minute and try again."), 503


if __name__ == "__main__":
    app.run(debug=True, port=5001)
