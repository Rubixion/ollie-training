FROM python:3.11-slim

# build-essential: insightface compiles a small C extension; libgl1/libglib2.0-0: OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs Spaces as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Bake the InsightFace detector (~280 MB download) into the image so cold starts don't fetch it
RUN python -c "from face_features import _get_insight_app; _get_insight_app()"

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
