from __future__ import annotations

import logging
import sys
from pathlib import Path

from pydantic import ValidationError


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def main() -> int:
    _configure_logging()
    _ensure_repo_on_path()

    logger = logging.getLogger("openrouter_smoke")

    try:
        from app.config import get_settings
        from app.llm import LLMError, generate_answer

        settings = get_settings()
        logger.info("Loaded settings; using model=%s", settings.model_name)

        prompt = "Reply with one short sentence confirming connectivity to OpenRouter."  # sample prompt
        response = generate_answer(prompt)
        print(response)
        return 0

    except ValidationError as exc:
        logger.error("Configuration error: %s", exc)
        logger.info("Tip: create a .env file from .env.example")
        return 2

    except LLMError as exc:
        logger.error("LLM error: %s", exc)
        return 3

    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130

    except Exception:
        logger.exception("Unexpected error")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
