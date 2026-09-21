import os
import time
from collections import defaultdict, deque

from flask import Flask, jsonify, render_template, request
from openai import OpenAI

app = Flask(__name__)

# ============================================================
# SETTINGS
# ============================================================

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

MAX_MESSAGE_LENGTH = 4000
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS = 6000

RATE_LIMIT_REQUESTS = 15
RATE_LIMIT_WINDOW = 60

API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY) if API_KEY else None


# ============================================================
# RATE LIMITER
# ============================================================

request_log = defaultdict(deque)


def is_rate_limited(ip):
    now = time.time()
    history = request_log[ip]

    while history and now - history[0] > RATE_LIMIT_WINDOW:
        history.popleft()

    if len(history) >= RATE_LIMIT_REQUESTS:
        return True

    history.append(now)
    return False


# ============================================================
# MODES
# ============================================================

MODE_INSTRUCTIONS = {
    "General": """
You are PyBot AI, a helpful general-purpose assistant.
Give clear, accurate and useful answers.
Keep answers reasonably concise.
""",

    "Study": """
You are PyBot AI in Study Mode.
Help students understand school subjects.
Give correct, simple and clear explanations.
Use points when useful.
""",

    "Coding": """
You are PyBot AI in Coding Mode.
Help with Python, Arduino, HTML, CSS, JavaScript,
Flask and other programming topics.
When fixing code, provide complete copy-paste-ready code.
""",

    "Creative": """
You are PyBot AI in Creative Mode.
Help with writing, ideas, captions and creative projects.
Keep responses clear and organized.
"""
}


# ============================================================
# CLEAN HISTORY
# ============================================================

def prepare_history(history):

    if not isinstance(history, list):
        return []

    result = []
    total_chars = 0

    # Only the latest few messages
    for item in history[-MAX_HISTORY_MESSAGES:]:

        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in ("user", "assistant"):
            continue

        if not isinstance(content, str):
            continue

        content = content.strip()

        if not content:
            continue

        # Limit each old message
        content = content[:1200]

        if total_chars + len(content) > MAX_HISTORY_CHARS:
            break

        result.append({
            "role": role,
            "content": content
        })

        total_chars += len(content)

    return result


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "service": "PyBotAI",
        "model": MODEL,
        "api_key_configured": bool(API_KEY)
    })


# ============================================================
# CHAT API
# ============================================================

@app.route("/api/chat", methods=["POST"])
def chat():

    # --------------------------------------------------------
    # IP
    # --------------------------------------------------------

    ip = request.headers.get(
        "X-Forwarded-For",
        request.remote_addr or "unknown"
    )

    if "," in ip:
        ip = ip.split(",")[0].strip()

    # --------------------------------------------------------
    # Rate limit
    # --------------------------------------------------------

    if is_rate_limited(ip):

        return jsonify({
            "error": "Too many requests. Please wait a moment and try again."
        }), 429

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    data = request.get_json(silent=True)

    if not isinstance(data, dict):

        return jsonify({
            "error": "Invalid request."
        }), 400

    # --------------------------------------------------------
    # Message
    # --------------------------------------------------------

    message = data.get("message", "")

    if not isinstance(message, str):

        return jsonify({
            "error": "Message must be text."
        }), 400

    message = message.strip()

    if not message:

        return jsonify({
            "error": "Please enter a message."
        }), 400

    if len(message) > MAX_MESSAGE_LENGTH:

        return jsonify({
            "error": "Message is too long."
        }), 400

    # --------------------------------------------------------
    # Mode
    # --------------------------------------------------------

    mode = data.get("mode", "General")

    if mode not in MODE_INSTRUCTIONS:
        mode = "General"

    # --------------------------------------------------------
    # Web Search
    # --------------------------------------------------------

    web_search = bool(data.get("web_search", False))

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = prepare_history(
        data.get("history", [])
    )

    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    if not API_KEY or client is None:

        return jsonify({
            "error": "OpenAI API key is not configured on Render."
        }), 500

    # --------------------------------------------------------
    # Instructions
    # --------------------------------------------------------

    instructions = MODE_INSTRUCTIONS[mode]

    instructions += """

Important rules:
- Answer the user's question directly.
- Do not unnecessarily repeat the question.
- Keep normal answers concise.
- Use Markdown when useful.
"""

    # --------------------------------------------------------
    # Build input
    # --------------------------------------------------------

    api_input = []

    for item in history:

        api_input.append({
            "role": item["role"],
            "content": item["content"]
        })

    api_input.append({
        "role": "user",
        "content": message
    })

    # --------------------------------------------------------
    # OPENAI REQUEST
    # --------------------------------------------------------

    try:

        if web_search:

            response = client.responses.create(
                model=MODEL,
                instructions=instructions,
                input=api_input,
                tools=[
                tools=[
    {
        "type": "web_search",
        "search_context_size": "low"
    }
],
                max_output_tokens=800
            )

        else:

            response = client.responses.create(
                model=MODEL,
                instructions=instructions,
                input=api_input,
                max_output_tokens=800
            )

        # ----------------------------------------------------
        # RESPONSE TEXT
        # ----------------------------------------------------

        reply = getattr(
            response,
            "output_text",
            None
        )

        if not reply:

            reply = "Sorry, I could not generate a response."

        return jsonify({
            "reply": reply,
            "mode": mode
        })

    # --------------------------------------------------------
    # ERRORS
    # --------------------------------------------------------

    except Exception as error:

        print("=" * 60)
        print("OPENAI ERROR")
        print(type(error).__name__)
        print(str(error))
        print("=" * 60)

        error_text = str(error).lower()

        if "rate_limit" in error_text:

            return jsonify({
                "error": (
                    "OpenAI rate limit reached. "
                    "Please wait a moment and try again."
                )
            }), 429

        if "authentication" in error_text or "api key" in error_text:

            return jsonify({
                "error": (
                    "OpenAI authentication failed. "
                    "Check OPENAI_API_KEY in Render."
                )
            }), 500

        if "model" in error_text and (
            "not found" in error_text
            or "unsupported" in error_text
        ):

            return jsonify({
                "error": (
                    "The selected OpenAI model is not available."
                )
            }), 500

        return jsonify({
            "error": "PyBot could not contact the AI service."
        }), 500


# ============================================================
# ERROR PAGES
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({
        "error": "Page not found."
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):

    return jsonify({
        "error": "Method not allowed."
    }), 405


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
