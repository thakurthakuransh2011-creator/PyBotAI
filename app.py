import os
import time
from collections import defaultdict

from flask import Flask, jsonify, render_template, request
from openai import OpenAI


# ============================================================
# PYBOT AI WEB v4.0
# ============================================================

app = Flask(__name__)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is missing."
    )

client = OpenAI(api_key=API_KEY)


# ------------------------------------------------------------
# AI MODES
# ------------------------------------------------------------

MODE_INSTRUCTIONS = {

    "General": """
You are PyBot AI, a helpful general-purpose AI assistant.

Give accurate, clear and useful answers.
Use simple language when appropriate.
Do not make answers unnecessarily long.
""",

    "Study": """
You are PyBot AI in Study Mode.

Help students understand their subjects.
Give correct, clear and age-appropriate explanations.
For school questions, provide direct answers with brief explanations.
Use points and examples when useful.
Do not make answers unnecessarily complicated.
""",

    "Coding": """
You are PyBot AI in Coding Mode.

You are an expert programming assistant.

Help with Python, Arduino, HTML, CSS, JavaScript and other
programming languages.

When the user asks for code:
- Give copy-paste-ready code.
- Check syntax carefully.
- Explain important parts briefly.
- Give complete code when the user requests complete code.
""",

    "Creative": """
You are PyBot AI in Creative Mode.

Help with creative writing, ideas, stories, project ideas,
presentations, names and imaginative content.

Be creative, useful and clear.
"""
}


# ------------------------------------------------------------
# BASIC PUBLIC-SERVER PROTECTION
# ------------------------------------------------------------

# Maximum characters accepted from one message.
MAX_MESSAGE_LENGTH = 6000

# Simple in-memory request limiter.
# This is not a replacement for a production rate-limit service,
# but helps prevent accidental excessive API requests.
REQUEST_LIMIT = 20
REQUEST_WINDOW = 60

request_log = defaultdict(list)


def check_rate_limit(ip_address):

    current_time = time.time()

    # Remove old requests
    request_log[ip_address] = [
        timestamp
        for timestamp in request_log[ip_address]
        if current_time - timestamp < REQUEST_WINDOW
    ]

    if len(request_log[ip_address]) >= REQUEST_LIMIT:
        return False

    request_log[ip_address].append(current_time)

    return True


# ------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------

@app.route("/")
def home():
    return render_template(
        "index.html",
        model=MODEL
    )


# ------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------

@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "app": "PyBot AI",
        "version": "4.0"
    })


# ------------------------------------------------------------
# CHAT API
# ------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def chat():

    # --------------------------------------------------------
    # Rate limit
    # --------------------------------------------------------

    client_ip = request.headers.get(
        "X-Forwarded-For",
        request.remote_addr
    )

    if client_ip and "," in client_ip:
        client_ip = client_ip.split(",")[0].strip()

    if not client_ip:
        client_ip = "unknown"

    if not check_rate_limit(client_ip):

        return jsonify({
            "error": "Too many requests. Please wait a little and try again."
        }), 429


    # --------------------------------------------------------
    # JSON data
    # --------------------------------------------------------

    data = request.get_json(silent=True)

    if not data:

        return jsonify({
            "error": "Invalid request."
        }), 400


    message = str(
        data.get("message", "")
    ).strip()

    mode = str(
        data.get("mode", "General")
    )

    web_search = bool(
        data.get("web_search", False)
    )

    history = data.get(
        "history",
        []
    )


    # --------------------------------------------------------
    # Validate message
    # --------------------------------------------------------

    if not message:

        return jsonify({
            "error": "Please enter a message."
        }), 400


    if len(message) > MAX_MESSAGE_LENGTH:

        return jsonify({
            "error": (
                f"Message is too long. "
                f"Maximum {MAX_MESSAGE_LENGTH} characters."
            )
        }), 400


    # --------------------------------------------------------
    # Validate mode
    # --------------------------------------------------------

    if mode not in MODE_INSTRUCTIONS:

        mode = "General"


    # --------------------------------------------------------
    # Build conversation
    # --------------------------------------------------------

    instructions = MODE_INSTRUCTIONS[mode]

    api_input = []


    # --------------------------------------------------------
    # Add previous messages
    # --------------------------------------------------------

    if isinstance(history, list):

        for item in history[-20:]:

            if not isinstance(item, dict):
                continue

            role = item.get("role")
            content = item.get("content")

            if role not in ["user", "assistant"]:
                continue

            if not isinstance(content, str):
                continue

            if not content.strip():
                continue

            api_input.append({
                "role": role,
                "content": content[:6000]
            })


    # --------------------------------------------------------
    # Add current message
    # --------------------------------------------------------

    api_input.append({
        "role": "user",
        "content": message
    })


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
                        "type": "web_search"
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
        # Get response text
        # ----------------------------------------------------

        answer = response.output_text

        if not answer:

            answer = (
                "Sorry, I couldn't generate a response."
            )


        return jsonify({
            "success": True,
            "answer": answer,
            "mode": mode,
            "web_search": web_search
        })


    except Exception as error:

        print("OpenAI API Error:", error)

        return jsonify({
            "success": False,
            "error": (
                "PyBot couldn't contact the AI service. "
                "Please try again."
            )
        }), 500


# ------------------------------------------------------------
# RUN LOCAL SERVER
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