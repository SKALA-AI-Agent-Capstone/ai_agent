import os
import secrets


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value or "__GENERATE_" in value:
        raise RuntimeError(f"{name} is missing. Run scripts/bootstrap.py and load .env.")
    return value


def internal_token() -> str:
    value = required_env("INTERNAL_SERVICE_TOKEN")
    if len(value) < 32:
        raise RuntimeError("INTERNAL_SERVICE_TOKEN must contain at least 32 characters.")
    return value


def valid_internal_authorization(header: str | None) -> bool:
    expected = f"Bearer {internal_token()}"
    return secrets.compare_digest(header or "", expected)
