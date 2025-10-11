# Serving Notes

The recommended serving path is vLLM (see the project README). If you expose the optional FastAPI app in `scripts/api.py`, run it behind an authenticating reverse proxy and add rate limiting (e.g., `--rate-limit 30` in nginx) to guard GPU resources.
