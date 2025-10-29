FROM python:3.10

ARG WWWUSER=1000
ARG WWWGROUP=1000

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --upgrade google-genai

RUN groupadd -g ${WWWGROUP} appgroup || true && \
    useradd -m -u ${WWWUSER} -g ${WWWGROUP} -s /bin/bash appuser || true

COPY . .

COPY --chown=appuser:appgroup . .

USER appuser
ENV HOME=/home/appuser

RUN chmod +x docker-entrypoint.sh
ENTRYPOINT ["./docker-entrypoint.sh"]
