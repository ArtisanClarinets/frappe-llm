#!/usr/bin/env bash
set -euo pipefail
uvicorn serve.api:app --host 0.0.0.0 --port 8000
