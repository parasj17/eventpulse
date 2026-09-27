# EventPulse, as a single container.
#
# The data is synthetic and deterministic, so it is generated and the model
# trained at build time rather than on every boot. That keeps startup instant,
# which matters when a judge clicks the link and waits.

FROM python:3.11-slim

WORKDIR /app

# Dependencies first, so code edits do not invalidate the layer.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY frontend ./frontend

# Bake the dataset and the trained model into the image.
WORKDIR /app/backend
RUN python generate_city.py && python -m engine.model

# Hugging Face Spaces expects 7860. Render and Fly pass $PORT instead, and the
# fallback covers both.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860}"]
