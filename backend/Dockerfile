# Build from repository root: docker build -f backend/Dockerfile -t aep-compute .
FROM node:24-bookworm-slim AS node
FROM python:3.10-slim-bookworm
COPY --from=node /usr/local/bin/node /usr/local/bin/node
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt \
    && useradd --create-home --uid 10001 appuser
COPY src/ /app/src/
COPY web/lib/ /app/web/lib/
COPY web/package.json /app/web/package.json
COPY backend/server.ts backend/package.json /app/backend/
ENV AEP_PYTHON=/usr/local/bin/python AEP_BIND_HOST=0.0.0.0 PORT=8000 PYTHONDONTWRITEBYTECODE=1
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD node -e 'fetch("http://127.0.0.1:"+process.env.PORT+"/healthz").then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))'
CMD ["node", "--disable-warning=ExperimentalWarning", "--experimental-strip-types", "backend/server.ts"]
