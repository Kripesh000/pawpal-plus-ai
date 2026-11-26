"""OPTIONAL multimodal second opinion (Anthropic Messages API).

Disabled unless the `anthropic` package is installed AND the ANTHROPIC_API_KEY environment variable
is set (see .env.example). It never replaces the heuristic analysis: its answer is schema-validated,
safety-filtered, shown separately, and used only to adjust confidence or to *raise* concern.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any, Dict, Optional, Sequence

from PIL import Image

from .imaging import LoadedImage
from .models import RetrievedReference
from .prompts import SYSTEM_PROMPT, build_user_prompt

DEFAULT_MODEL = "claude-sonnet-5-5"


class LLMUnavailable(RuntimeError):
    """Raised when the optional backend is not configured or the call fails."""


class LLMVisionBackend:
    """Thin wrapper that asks a multimodal model for a constrained JSON observation."""

    def __init__(self, client: Any = None, model: Optional[str] = None) -> None:
        self.model = model or os.environ.get("PAWPAL_LLM_MODEL", DEFAULT_MODEL)
        self._client = client

    @staticmethod
    def is_configured() -> bool:
        """True when the SDK is importable and an API key is present in the environment."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.is_configured():
            raise LLMUnavailable("ANTHROPIC_API_KEY not set or the 'anthropic' package is not installed.")
        import anthropic

        self._client = anthropic.Anthropic()
        return self._client

    def describe(self, image: LoadedImage, species: str, references: Sequence[RetrievedReference],
                 history_summary: str) -> Dict[str, Any]:
        """Return the model's parsed JSON answer (unvalidated). Raises LLMUnavailable on any failure."""
        client = self._get_client()
        buf = io.BytesIO()
        Image.fromarray(image.rgb).save(buf, format="PNG")
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.b64encode(buf.getvalue()).decode("ascii")}},
            {"type": "text", "text": build_user_prompt(species, references, history_summary)},
        ]
        try:
            response = client.messages.create(model=self.model, max_tokens=600, system=SYSTEM_PROMPT,
                                              messages=[{"role": "user", "content": content}])
            text = "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text")
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                raise ValueError("no JSON object in response")
            return json.loads(match.group(0))
        except LLMUnavailable:
            raise
        except Exception as exc:  # network, auth, parse errors all degrade to "no second opinion"
            raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc
