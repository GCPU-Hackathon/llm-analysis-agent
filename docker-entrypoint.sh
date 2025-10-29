#!/bin/bash
set -e

SSL_CERT_PATH="/app/ssl/certificate.crt"
SSL_KEY_PATH="/app/ssl/private.key"

if [ -f "$SSL_CERT_PATH" ] && [ -f "$SSL_KEY_PATH" ]; then
    echo "SSL certificates found. Starting server with HTTPS support..."
    exec uvicorn main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --ssl-keyfile "$SSL_KEY_PATH" \
        --ssl-certfile "$SSL_CERT_PATH" \
        --reload
else
    echo "SSL certificates not found at $SSL_CERT_PATH and $SSL_KEY_PATH"
    echo "Starting server with HTTP only..."
    exec uvicorn main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload
fi
