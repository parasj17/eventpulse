"""
Vercel entry point: the EventPulse FastAPI backend as one Python serverless function.

vercel.json sends every API path (/zones, /auth/..., /organizer/..., /status/... and so on)
here; the static pages in frontend/ are served directly by Vercel's CDN.
"""
import os
import sys

BACKEND = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
sys.path.insert(0, BACKEND)

from api import app  # noqa: E402  (backend/api.py, found first on the path above)
