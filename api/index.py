"""
api/index.py — Vercel entry point for Dokai Healthcare AI.

IMPORTANT: This file must live inside an `api/` folder in your repo root.
The repo structure must be:
  dokai-beta/
    api/
      index.py          ← this file
    app.py              ← the main Flask app
    requirements.txt    ← clean, no comments
    vercel.json         ← deployment config

Vercel calls `handler` as the WSGI entry point.
Tesseract is NOT available on Vercel — the app auto-falls back to OCR.space API.
Set OCR_SPACE_API_KEY in Vercel → Settings → Environment Variables.
"""
import sys
import os

# Make sure app.py (one level up from api/) is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app

# Vercel expects a variable named `handler`
handler = app
