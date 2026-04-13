"""
NVIDIA LLM Bridge API
OpenAI-compatible Flask server that proxies requests to NVIDIA's inference API.
Deployable on Render.com with gunicorn.
"""

import os
import uuid
import time
import logging
import requests
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

# ─────────────────────────────────────────────
#  App & Logging Setup
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
NVIDIA_API_KEY   = os.environ.get("NVIDIA_API_KEY")
NVIDIA_BASE_URL  = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL    = "meta/llama-3.1-70b-instruct"
REQUEST_TIMEOUT  = 60          # seconds – streaming needs more headroom
MAX_TOKENS       = 2048
TEMPERATURE      = 0.2
TOP_P            = 0.7

SYSTEM_PROMPT = """\
You are a senior Android developer with 10+ years of real-world experience \
building production-grade Android applications.

Behave exactly like a human expert developer would:
- Think step-by-step before writing any code.
- Always produce complete, functional, copy-paste-ready code — no placeholders, \
  no "TODO" comments, no pseudo-code.
- Use Java by default unless the user explicitly asks for Kotlin.
- Follow Google's official Android coding conventions and Material Design guidelines.
- Add concise but meaningful inline comments where non-obvious.
- Structure every file correctly: package declaration, imports, class, \
  lifecycle methods, then helpers.
- Avoid deprecated APIs; target the latest stable Android SDK.
- When answering questions (not code), give direct, professional answers — \
  the way a senior colleague on Slack would reply, not a textbook.
- If a request is ambiguous, make a reasonable assumption, state it briefly, \
  then deliver the solution.
"""

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _error_response(message: str, status: int = 500):
    """Always return a valid OpenAI-shaped error so clients never crash."""
    logger.error("Returning error to client: %s", message)
    body = {
        "id": f"chatcmpl-err-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": DEFAULT_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Error: {message}",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    return jsonify(body), status


def _validate_key():
    if not NVIDIA_API_KEY:
        return "NVIDIA_API_KEY environment variable is not set."
    return None


def _build_messages(client_messages: list) -> list:
    """Prepend the system prompt, then attach client messages."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in client_messages:
        role    = str(m.get("role", "user")).strip()
        content = str(m.get("content", "")).strip()
        if role not in {"system", "user", "assistant"}:
            role = "user"
        # Don't duplicate a system message if the client already sent one
        if role == "system":
            continue
        messages.append({"role": role, "content": content})
    return messages


def _nvidia_headers() -> dict:
    return {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health_check():
    """Render health-check & quick status endpoint."""
    key_ok = bool(NVIDIA_API_KEY)
    return jsonify({
        "status": "ok",
        "service": "NVIDIA LLM Bridge",
        "nvidia_key_loaded": key_ok,
        "default_model": DEFAULT_MODEL,
        "timestamp": int(time.time()),
    }), 200


@app.route("/v1/models", methods=["GET"])
def list_models():
    """Stub so OpenAI-compatible clients don't break on model discovery."""
    return jsonify({
        "object": "list",
        "data": [
            {"id": DEFAULT_MODEL, "object": "model", "owned_by": "nvidia"},
            {"id": "meta/llama-3.1-8b-instruct",  "object": "model", "owned_by": "nvidia"},
            {"id": "mistralai/mixtral-8x7b-instruct-v0.1", "object": "model", "owned_by": "nvidia"},
        ],
    }), 200


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    # ── 1. API key guard ────────────────────────────────────────────────────
    key_err = _validate_key()
    if key_err:
        return _error_response(key_err, 503)

    # ── 2. Parse & validate request body ───────────────────────────────────
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return _error_response("Request body must be valid JSON.", 400)
    except Exception:
        return _error_response("Failed to parse request body.", 400)

    client_messages = data.get("messages")
    if not isinstance(client_messages, list) or not client_messages:
        return _error_response("'messages' must be a non-empty array.", 400)

    model   = data.get("model") or DEFAULT_MODEL
    stream  = bool(data.get("stream", False))

    logger.info(
        "Request | model=%s stream=%s messages=%d",
        model, stream, len(client_messages),
    )

    # ── 3. Build NVIDIA payload ─────────────────────────────────────────────
    messages = _build_messages(client_messages)

    payload = {
        "model":       DEFAULT_MODEL,   # always use the pinned capable model
        "messages":    messages,
        "temperature": float(data.get("temperature", TEMPERATURE)),
        "top_p":       float(data.get("top_p", TOP_P)),
        "max_tokens":  int(data.get("max_tokens", MAX_TOKENS)),
        "stream":      stream,
    }

    # ── 4a. Streaming path ──────────────────────────────────────────────────
    if stream:
        def generate():
            try:
                with requests.post(
                    NVIDIA_BASE_URL,
                    headers=_nvidia_headers(),
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                    stream=True,
                ) as resp:
                    if resp.status_code != 200:
                        err_chunk = (
                            'data: {"choices":[{"delta":{"content":'
                            f'"Error: NVIDIA API returned {resp.status_code}"}}'
                            ']}\n\ndata: [DONE]\n\n'
                        )
                        yield err_chunk
                        return
                    for line in resp.iter_lines():
                        if line:
                            yield line.decode("utf-8") + "\n\n"
            except requests.exceptions.Timeout:
                yield (
                    'data: {"choices":[{"delta":{"content":'
                    '"Error: Request timed out."}}]}\n\ndata: [DONE]\n\n'
                )
            except Exception as exc:
                yield (
                    f'data: {{"choices":[{{"delta":{{"content":'
                    f'"Error: {str(exc)}"}}}}]}}\n\ndata: [DONE]\n\n'
                )

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ── 4b. Non-streaming path ──────────────────────────────────────────────
    try:
        resp = requests.post(
            NVIDIA_BASE_URL,
            headers=_nvidia_headers(),
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.Timeout:
        return _error_response("NVIDIA API request timed out.", 504)
    except requests.exceptions.ConnectionError as exc:
        return _error_response(f"Could not reach NVIDIA API: {exc}", 502)
    except Exception as exc:
        return _error_response(f"Unexpected network error: {exc}", 500)

    # ── 5. Validate NVIDIA response ─────────────────────────────────────────
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail") or resp.text[:200]
        except Exception:
            detail = resp.text[:200]
        logger.error("NVIDIA API error %d: %s", resp.status_code, detail)
        return _error_response(f"NVIDIA API error {resp.status_code}: {detail}", 502)

    try:
        nvidia_data = resp.json()
    except Exception:
        return _error_response("NVIDIA returned non-JSON response.", 502)

    choices = nvidia_data.get("choices")
    if not choices or not isinstance(choices, list):
        return _error_response("NVIDIA response missing 'choices'.", 502)

    # ── 6. Build OpenAI-compatible response ─────────────────────────────────
    first   = choices[0]
    content = (
        first.get("message", {}).get("content")
        or first.get("delta", {}).get("content")
        or ""
    )

    response_body = {
        "id":      nvidia_data.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}"),
        "object":  "chat.completion",
        "created": nvidia_data.get("created", int(time.time())),
        "model":   nvidia_data.get("model", DEFAULT_MODEL),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role":    "assistant",
                    "content": content,
                },
                "finish_reason": first.get("finish_reason", "stop"),
            }
        ],
        "usage": nvidia_data.get("usage", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }),
    }

    logger.info(
        "OK | finish=%s tokens=%s",
        first.get("finish_reason"),
        nvidia_data.get("usage", {}).get("total_tokens", "?"),
    )
    return jsonify(response_body), 200


# ─────────────────────────────────────────────
#  Entry point (dev only — prod uses gunicorn)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting dev server on port %d (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)
