"""StyleMate: AI dress code agent (Flask + Gemini) with shopping links.
Run with: python app.py
"""
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from google import genai
from google.genai import types

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "").strip()
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "").strip()  # free key from pixabay.com/api/docs
RETRIES = 4  # attempts per model when Google says "busy"
SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text(encoding="utf-8")
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Added to the system prompt so the model also returns a shopping list the app can turn into links.
JSON_RULES = """

OUTPUT FORMAT OVERRIDE (this replaces any other output format above):
Reply with ONLY one JSON object, no code fences, shaped exactly like this:
{"advice": "<your full answer as Markdown, in the format described above>",
 "items": [{"for": "Boy or Girl or Person", "name": "<short product name>",
            "search_query": "<what to type in a shopping site search box, e.g. men emerald green silk blend kurta>",
            "price_min_inr": 1200, "price_max_inr": 2500}]}
List 3 to 6 individual wearable items from your outfit (one entry per item, per person if the
request is for both). Prices are your rough estimates in Indian rupees (numbers only).
Never invent store names, brands or web links."""

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024  # 6 MB upload limit


def is_busy_error(e):
    """True for temporary Google-side problems (503 overloaded, 500, 429 rate limit)."""
    code = getattr(e, "code", None)
    return code in (429, 500, 502, 503, 504) or "UNAVAILABLE" in str(e)


def build_links(query):
    """Search links for each shop. No scraping and no paid keys: just URLs."""
    q = quote_plus(query)
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
    return [
        {"site": "Amazon", "url": f"https://www.amazon.in/s?k={q}"},
        {"site": "Flipkart", "url": f"https://www.flipkart.com/search?q={q}"},
        {"site": "Myntra", "url": f"https://www.myntra.com/{slug}"},
        {"site": "Ajio", "url": f"https://www.ajio.com/search/?text={q}"},
        {"site": "Meesho", "url": f"https://www.meesho.com/search?q={q}"},
        {"site": "Compare on Google Shopping", "url": f"https://www.google.com/search?tbm=shop&q={q}"},
    ]


def to_int(v):
    try:
        return int(float(str(v).replace(",", "").strip()))
    except (ValueError, TypeError):
        return None


@lru_cache(maxsize=256)
def fetch_photo(query):
    """Find one free stock photo for a search phrase using the Pixabay API.
    Returns a dict, or None if there is no key, no result, or any problem."""
    if not PIXABAY_API_KEY:
        return None
    try:
        url = ("https://pixabay.com/api/?image_type=photo&orientation=vertical&safesearch=true"
               "&per_page=3&key=" + quote_plus(PIXABAY_API_KEY) + "&q=" + quote_plus(query[:100]))
        req = urllib.request.Request(url, headers={"User-Agent": "StyleMate/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        hits = data.get("hits") or []
        if not hits:
            return None
        p = hits[0]
        img = p.get("webformatURL", "")
        page = p.get("pageURL", "")
        # Only accept real Pixabay addresses before sending them to the browser.
        ok_img = img.startswith(("https://pixabay.com/", "https://cdn.pixabay.com/"))
        if not ok_img or not page.startswith("https://pixabay.com/"):
            return None
        return {
            "image": img,
            "page": page,
            "photographer": str(p.get("user", ""))[:60],
            "alt": str(p.get("tags") or query)[:120],
        }
    except Exception as e:  # photos are a bonus: never break the whole answer
        app.logger.warning("Pixabay lookup failed for %r: %s", query, type(e).__name__)
        return None


def clean_items(raw):
    items = []
    for it in (raw or [])[:6]:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name", "")).strip()[:80]
        query = str(it.get("search_query") or name).strip()[:120]
        if not name or not query:
            continue
        lo, hi = to_int(it.get("price_min_inr")), to_int(it.get("price_max_inr"))
        price_text = ""
        if lo and hi:
            price_text = f"approx. Rs {lo:,} - Rs {hi:,}"
        elif lo or hi:
            price_text = f"approx. Rs {(lo or hi):,}"
        items.append({
            "for": str(it.get("for", "")).strip()[:20],
            "name": name,
            "price_text": price_text,
            "links": build_links(query),
            "_query": query,
        })
    # Look up all photos at the same time so the page stays fast.
    if items and PIXABAY_API_KEY:
        with ThreadPoolExecutor(max_workers=6) as pool:
            photos = list(pool.map(lambda i: fetch_photo(i["_query"]), items))
        for item, photo in zip(items, photos):
            item["photo"] = photo
    for item in items:
        item.pop("_query", None)
    return items


def parse_reply(text):
    """Return (advice_markdown, items). Falls back to plain text if the JSON is broken."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1]):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and data.get("advice"):
                return str(data["advice"]), clean_items(data.get("items"))
        except (ValueError, TypeError):
            continue
    return text or "No answer came back. Please try again.", []


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
  .note{color:#666;font-size:.9em;margin:4px 0 12px}
  .card{background:#fff;border:1px solid #e3e0ff;border-radius:10px;padding:12px;margin:10px 0}
  .links{margin-top:8px;display:flex;flex-wrap:wrap;gap:8px}
  .links a{padding:6px 12px;border-radius:999px;background:#ece9ff;color:#3b22c9;text-decoration:none;font-size:.9em}
  .links a:hover{background:#d9d3ff}
  .item{display:flex;gap:14px;align-items:flex-start}
  .item img{width:110px;height:140px;object-fit:cover;border-radius:8px;flex:none;background:#eee}
  .credit{font-size:.75em;color:#777;margin-top:4px}
  .credit a{color:#777}
</style></head><body>
<h1>StyleMate</h1><p class="sub">Get an outfit suggestion, with or without a photo, plus shopping links.</p>
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

function renderItems(items){
  if(!items||!items.length) return;
  const h=document.createElement('h3'); h.textContent='Shop these items'; out.appendChild(h);
  const note=document.createElement('p'); note.className='note';
  note.textContent='Prices are rough estimates. Tap a store to see live prices, or use Google Shopping to compare many stores at once.';
  out.appendChild(note);
  let anyPhoto=false;
  items.forEach(it=>{
    const card=document.createElement('div'); card.className='card item';
    if(it.photo){
      anyPhoto=true;
      const img=document.createElement('img'); img.src=it.photo.image; img.alt=it.photo.alt; img.loading='lazy';
      card.appendChild(img);
    }
    const body=document.createElement('div');
    const t=document.createElement('strong');
    t.textContent=(it.for?'['+it.for+'] ':'')+it.name+(it.price_text?' - '+it.price_text:'');
    body.appendChild(t);
    const row=document.createElement('div'); row.className='links';
    it.links.forEach(l=>{
      const a=document.createElement('a'); a.href=l.url; a.target='_blank'; a.rel='noopener noreferrer';
      a.textContent=l.site; row.appendChild(a);
    });
    body.appendChild(row);
    if(it.photo){
      const c=document.createElement('div'); c.className='credit';
      c.appendChild(document.createTextNode('Similar style. Photo by '+(it.photo.photographer||'a photographer')+' on '));
      const pa=document.createElement('a'); pa.href=it.photo.page; pa.target='_blank'; pa.rel='noopener noreferrer';
      pa.textContent='Pixabay'; c.appendChild(pa); body.appendChild(c);
    }
    card.appendChild(body); out.appendChild(card);
  });
}

f.addEventListener('submit',async e=>{
  e.preventDefault(); b.disabled=true; b.textContent='Thinking...';
  out.style.display='block'; out.textContent='Working on your outfit...';
  try{
    const r=await fetch('/suggest',{method:'POST',body:new FormData(f)});
    const d=await r.json();
    if(d.error){out.innerHTML='<span class="err"></span>';out.firstChild.textContent=d.error;}
    else{
      out.innerHTML=(window.marked?marked.parse(d.result):d.result.replace(/\\n/g,'<br>'));
      renderItems(d.items);
    }
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

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT + JSON_RULES,
        temperature=0.7,
        response_mime_type="application/json",
    )
    # Try the main model, then the optional backup model from .env (GEMINI_FALLBACK_MODEL).
    models = [MODEL] + ([FALLBACK_MODEL] if FALLBACK_MODEL and FALLBACK_MODEL != MODEL else [])
    for model in models:
        for attempt in range(RETRIES):
            try:
                resp = client.models.generate_content(model=model, contents=parts, config=config)
                advice, items = parse_reply(resp.text)
                return jsonify(result=advice, items=items)
            except Exception as e:
                app.logger.warning("Gemini %s attempt %d failed: %s", model, attempt + 1, e)
                if not is_busy_error(e):
                    return jsonify(error=f"AI error: {e}"), 502  # real problem: retrying won't help
                if attempt < RETRIES - 1:
                    time.sleep(2 ** (attempt + 1))  # wait 2s, 4s, 8s...

    return jsonify(error="Google's AI model is very busy right now. "
                         "Please wait a minute and try again."), 503


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1", port=int(os.getenv("PORT", "5001")))
