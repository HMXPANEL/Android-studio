import os
import uuid
import time
import logging
import requests
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ───────── CONFIG ─────────
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

DEFAULT_MODEL = "meta/llama-3.1-70b-instruct"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 2

SYSTEM_PROMPT = """You are an expert Android developer. 
Always return clean, production-ready Android code (Java/Kotlin)."""

# ───────── HELPERS ─────────

def _error(message, status=500):
    return jsonify({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": f"Error: {message}"
            }
        }]
    }), status


def _headers():
    return {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }


def _build_messages(client_msgs):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in client_msgs:
        role = m.get("role", "user")
        if role != "system":
            msgs.append(m)
    return msgs


# ───────── ROUTES ─────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "nvidia_key_loaded": bool(NVIDIA_API_KEY)
    })


@app.route("/v1/chat/completions", methods=["POST"])
def chat():

    if not NVIDIA_API_KEY:
        return _error("API key missing", 503)

    data = request.get_json(silent=True)
    if not data:
        return _error("Invalid JSON", 400)

    messages = data.get("messages")
    if not messages:
        return _error("Messages missing", 400)

    model = data.get("model", DEFAULT_MODEL)

    payload = {
        "model": model,
        "messages": _build_messages(messages),
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024
    }

    # ───────── RETRY LOGIC ─────────
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                NVIDIA_URL,
                headers=_headers(),
                json=payload,
                timeout=REQUEST_TIMEOUT
            )

            # 🔴 RATE LIMIT
            if resp.status_code == 429:
                logger.warning("Rate limit hit")
                time.sleep(2)
                continue

            # 🔴 OTHER ERROR
            if resp.status_code != 200:
                return _error(f"NVIDIA error {resp.status_code}", 502)

            # 🔴 JSON SAFETY
            try:
                data = resp.json()
            except:
                return _error("Invalid response (HTML or non-JSON)", 502)

            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            return jsonify({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content
                    }
                }]
            })

        except requests.exceptions.Timeout:
            logger.warning("Timeout retry...")
            time.sleep(1)
        except Exception as e:
            return _error(str(e), 500)

    return _error("Failed after retries (rate limit or timeout)", 500)


# ───────── RUN ─────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
