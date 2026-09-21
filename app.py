import os
import time
from collections import defaultdict, deque

from flask import Flask, jsonify, render_template, request
from openai import OpenAI


# ============================================================
# PyBotAI - Flask + OpenAI Responses API
# ============================================================

app = Flask(__name__)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
MAX_MESSAGE_LENGTH = 6000
RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW = 60

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    print("WARNING: OPENAI_API_KEY is not set.")

client = OpenAI(api_key=API_KEY) if API_KEY else None


# ------------------------------------------------------------
# Rate limiter
# ------------------------------------------------------------

request_log = defaultdict(deque)


def is_rate_limited(ip_address):
    now = time.time()
    history = request_log[ip_address]

    while history and now - history[0] > RATE_LIMIT_WINDOW:
        history.popleft()

    if len(history) >= RATE_LIMIT_REQUESTS:
        return True

    history.append(now)
    return False


# ------------------------------------------------------------
# Modes
# ------------------------------------------------------------

MODE_INSTRUCTIONS = {
    "General": """
You are PyBot, a helpful AI assistant.

Answer clearly, accurately, and naturally.
Use simple language when possible.
If the user asks for steps, give numbered steps.
If the user asks for code, provide clean copy-paste-ready code.
""",

    "Study": """
You are PyBot in Study Mode.

Help the student understand their school subjects.
Give correct, clear, age-appropriate explanations.
For homework, give the answer first and then a short explanation when useful.
Prefer point-wise answers when appropriate.
""",

    "Coding": """
You are PyBot in Coding Mode.

Help with programming, debugging, Arduino, Python,
Flask, HTML, CSS, JavaScript, and related topics.

When fixing code:
- explain the problem briefly
- provide complete corrected code when practical
- make the code copy-paste-ready
- avoid unnecessary complexity
""",

    "Creative": """
You are PyBot in Creative Mode.

Help with creative writing, ideas, captions,
project ideas, descriptions, and brainstorming.

Keep the response useful and well organized.
"""
}


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def clean_mode(mode):
    if mode in MODE_INSTRUCTIONS:
        return mode
    return "General"


def clean_history(history):
    """
    Keep only a small amount of valid conversation history.
    """
    if not isinstance(history, list):
        return []

    cleaned = []

    for item in history[-20:]:
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

        content = content[:6000]

        cleaned.append({
            "role": role,
            "content": content
        })

    return cleaned


def build_input(history, message):
    """
    Build Responses API input.
    """

    messages = []

    for item in history:
        messages.append({
            "role": item["role"],
            "content": item["content"]
        })

    messages.append({
        "role": "user",
        "content": message
    })

    return messages


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "PyBotAI",
        "model": MODEL,
        "api_key_configured": bool(API_KEY)
    })


# ------------------------------------------------------------
# Chat API
# ------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def chat():

    # --------------------------------------------------------
    # Basic request validation
    # --------------------------------------------------------

    ip_address = request.headers.get(
        "X-Forwarded-For",
        request.remote_addr or "unknown"
    )

    if "," in ip_address:
        ip_address = ip_address.split(",")[0].strip()

    if is_rate_limited(ip_address):
        return jsonify({
            "error": "Too many requests. Please wait a little and try again."
        }), 429

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Invalid JSON request."
        }), 400

    message = data.get("message", "")
    mode = clean_mode(data.get("mode", "General"))
    web_search = bool(data.get("web_search", False))
    history = clean_history(data.get("history", []))

    # --------------------------------------------------------
    # Message validation
    # --------------------------------------------------------

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
            "error": f"Message is too long. Maximum {MAX_MESSAGE_LENGTH} characters."
        }), 400

    # --------------------------------------------------------
    # API key check
    # --------------------------------------------------------

    if not API_KEY or client is None:
        print("ERROR: OPENAI_API_KEY is missing.")

        return jsonify({
            "error": "OpenAI API key is not configured on the server."
        }), 500

    # --------------------------------------------------------
    # Instructions
    # --------------------------------------------------------

    instructions = MODE_INSTRUCTIONS[mode]

    instructions += """

Important:
- Do not claim that you performed an action you did not perform.
- If information may be current and web search is enabled, use web search.
- Keep answers readable.
- Use Markdown when it improves readability.
"""

    # --------------------------------------------------------
    # Build conversation
    # --------------------------------------------------------

    api_input = build_input(history, message)

    # --------------------------------------------------------
    # OpenAI request
    # --------------------------------------------------------

    try:

        if web_search:

            response = client.responses.create(
                model=MODEL,
                instructions=instructions,
                tools=[
                    {
                        "type": "web_search_preview"
                    }
                ],
                input=api_input
            )

        else:

            response = client.responses.create(
                model=MODEL,
                instructions=instructions,
                input=api_input
            )

        # ----------------------------------------------------
        # Extract final text
        # ----------------------------------------------------

        reply = getattr(response, "output_text", None)

        if not reply:

            # Fallback parser
            output_items = getattr(response, "output", []) or []

            collected_text = []

            for item in output_items:

                item_type = getattr(item, "type", "")

                if item_type == "message":

                    content_items = getattr(item, "content", []) or []

                    for content in content_items:

                        content_type = getattr(
                            content,
                            "type",
                            ""
                        )

                        if content_type in (
                            "output_text",
                            "text"
                        ):

                            text_value = getattr(
                                content,
                                "text",
                                ""
                            )

                            if text_value:
                                collected_text.append(
                                    text_value
                                )

            reply = "\n".join(collected_text).strip()

        if not reply:
            reply = "I received an empty response from the AI."

        return jsonify({
            "reply": reply,
            "mode": mode
        })

    # --------------------------------------------------------
    # OpenAI API errors
    # --------------------------------------------------------

    except Exception as error:

        print("=" * 70)
        print("OPENAI REQUEST ERROR")
        print(type(error).__name__)
        print(str(error))
        print("=" * 70)

        error_text = str(error)

        # Do not expose unnecessary internal information
        if "api key" in error_text.lower() or "authentication" in error_text.lower():
            user_error = (
                "OpenAI authentication failed. "
                "Please check OPENAI_API_KEY in Render Environment Variables."
            )

        elif "model" in error_text.lower() and (
            "not found" in error_text.lower()
            or "does not exist" in error_text.lower()
            or "unsupported" in error_text.lower()
        ):
            user_error = (
                f"The model '{MODEL}' is not available for this API project. "
                "Check the OPENAI_MODEL environment variable."
            )

        elif "web_search" in error_text.lower():
            user_error = (
                "Web Search is not available with the current API configuration. "
                "Try turning Web Search off."
            )

        elif "rate" in error_text.lower():
            user_error = (
                "The AI service rate limit was reached. "
                "Please wait a moment and try again."
            )

        else:
            user_error = (
                "PyBot could not contact the AI service. "
                "Check the Render logs for the exact error."
            )

        return jsonify({
            "error": user_error
        }), 500


# ------------------------------------------------------------
# Error handlers
# ------------------------------------------------------------

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


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error."
    }), 500


# ------------------------------------------------------------
# Local / Render server
# ------------------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
