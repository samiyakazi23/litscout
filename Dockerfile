FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /app
COPY requirements.txt requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements-server.txt
COPY --chown=user . .
RUN mkdir -p data/cache data/llm_cache data/public && chown -R user /app/data
USER user
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn litscout.server:app --host 0.0.0.0 --port ${PORT}"]
