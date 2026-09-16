from typing import Any

import httpx2

from .._globals import PROCESS_TIMEOUT
from ..http_client import http_client
from ..logging import xlog
from ..models import JaiMessage, JaiResult, JaiResultMetadata, JaiResultTokenUsage
from ..statistics import track_stats
from ..xuiduser import XUID


def vercel_generate_content(
    user: XUID,
    api_key: str,
    model: str,
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Wrapper around Vercel AI Gateway's Chat Completions API.

    User parameter is only used for logging.
    """

    vercel_request = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "content": message.content,
                "role": message.role,
            }
            for message in messages
        ],
    }

    for key, value in (settings or {}).items():
        if key == "temperature":
            vercel_request["temperature"] = value
        elif key == "max_tokens":
            vercel_request["max_tokens"] = value
        elif key == "top_p":
            vercel_request["top_p"] = value
        elif key == "frequency_penalty":
            vercel_request["frequency_penalty"] = value
        elif key == "repetition_penalty":
            vercel_request["presence_penalty"] = value

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        vercel_response = http_client.post(
            "https://ai-gateway.vercel.sh/v1/chat/completions",
            json=vercel_request,
            headers=headers,
            timeout=PROCESS_TIMEOUT,
        )

        vercel_response.raise_for_status()
        vercel_result = vercel_response.json()

    except httpx2.TimeoutException:
        track_stats("vercel.time_out")
        return JaiResult(504, "Vercel AI Gateway Timeout")

    except httpx2.HTTPStatusError as e:
        message = f"Error from Vercel AI Gateway ({e.response.status_code})"
        extras = ""

        try:
            error_result = e.response.json()
            error = error_result.get("error")

            if isinstance(error, dict):
                if error_message := error.get("message"):
                    message += f": {error_message}"
                extras = str(error)
            else:
                extras = str(error_result)

        except Exception:
            extras = e.response.text

        xlog(user, f"{message}: {extras!r}")

        if e.response.is_client_error:
            track_stats("vercel.failed.client")
        elif e.response.is_server_error:
            track_stats("vercel.failed.server")
        else:
            track_stats("vercel.failed.unknown")

        return JaiResult(
            e.response.status_code,
            message,
            extras=extras,
        )

    except Exception as e:  # ruff: ignore[BLE001]
        xlog(user, repr(e))
        track_stats("vercel.failed.exception")
        return JaiResult(502, "Unhandled exception from Vercel AI Gateway.")

    try:
        text = str(
            vercel_result["choices"][0]["message"]["content"] or ""
        )
    except (KeyError, IndexError, TypeError):
        text = ""

    metadata = JaiResultMetadata()

    if usage := vercel_result.get("usage"):
        metadata.token_usage = JaiResultTokenUsage(
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    if not text:
        xlog(user, f"No result text from Vercel: {vercel_result!r}")
        track_stats("vercel.rejected")
        return JaiResult(
            502,
            "Response blocked/empty.",
            metadata=metadata,
        )

    track_stats("vercel.succeeded")
    return JaiResult(200, text, metadata=metadata)
