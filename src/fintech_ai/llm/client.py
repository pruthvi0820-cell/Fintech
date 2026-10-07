"""Claude client wrapper.

Responsibilities: one place for model/effort config, the refusal-fallback opt-in,
error translation into domain exceptions, and stop_reason checks. Analysis features
depend on `ClaudeAnalyst`, never on the raw SDK, so it can be mocked in tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import anthropic
from anthropic.types.beta import BetaMessageParam, BetaOutputConfigParam

from fintech_ai.config import Settings
from fintech_ai.exceptions import LLMError
from fintech_ai.llm.prompts import TREND_ANALYST_SYSTEM, build_trend_prompt

logger = logging.getLogger(__name__)

# Server-side refusal fallback: if a safety classifier declines, the API re-runs the
# request on Anthropic's recommended fallback model inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_OUTPUT_TOKENS = 16_000


@dataclass(frozen=True)
class AnalysisResult:
    text: str
    model: str
    stop_reason: str | None
    truncated: bool
    input_tokens: int
    output_tokens: int
    request_id: str | None


class ClaudeAnalyst:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client or anthropic.Anthropic(
            timeout=settings.llm_timeout_seconds,
            max_retries=3,  # SDK backs off on 408/409/429/5xx and connection errors
        )

    def summarize_trend(self, snapshot: dict[str, Any]) -> AnalysisResult:
        if not snapshot:
            raise LLMError("Cannot summarize an empty snapshot")
        return self._complete(TREND_ANALYST_SYSTEM, build_trend_prompt(snapshot))

    def _complete(self, system: str, user_content: str) -> AnalysisResult:
        output_config = cast(BetaOutputConfigParam, {"effort": self._settings.anthropic_effort})
        messages: list[BetaMessageParam] = [{"role": "user", "content": user_content}]
        try:
            response = self._client.beta.messages.create(
                model=self._settings.anthropic_model,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=system,
                messages=messages,
                output_config=output_config,
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as exc:
            raise LLMError(
                "Anthropic authentication failed. Set ANTHROPIC_API_KEY or run `ant auth login`."
            ) from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMError("API key lacks permission for this model/feature.") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Model not found: {self._settings.anthropic_model!r}") from exc
        except anthropic.BadRequestError as exc:
            raise LLMError(f"Bad request to Claude API: {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Rate limited by Claude API after retries; try again later.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APITimeoutError as exc:
            raise LLMError("Claude API request timed out.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach Claude API (network error).") from exc

        return self._parse(response)

    @staticmethod
    def _parse(response: Any) -> AnalysisResult:
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMError(f"Model declined the request (category={category}).")

        blocks = response.content or []
        text = "".join(b.text for b in blocks if getattr(b, "type", None) == "text").strip()
        if not text:
            raise LLMError(f"Model returned no text (stop_reason={stop_reason}).")

        truncated = stop_reason == "max_tokens"
        if truncated:
            logger.warning("Claude output hit max_tokens and is truncated.")

        usage = getattr(response, "usage", None)
        return AnalysisResult(
            text=text,
            model=str(getattr(response, "model", "unknown")),
            stop_reason=stop_reason,
            truncated=truncated,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            request_id=getattr(response, "_request_id", None),
        )
