"""
app.py -- web version of the chatbot.

Serves a simple chat page (templates/index.html) and a /chat endpoint
that the page's JavaScript calls with each message. Uses the exact same
bot_core.get_bot_reply() as the terminal version -- nothing about the
classifier or the LLM fallback changes here.
"""
import flash
from flask import Flask, request, jsonify, render_template, session
from bot_core import get_bot_reply

app = Flask(__name__)

# Needed so Flask can keep a small per-visitor conversation history in a
# secure cookie. For a showcase project this is fine as-is; if you deploy
# this for real, set this from an environment variable instead.
app.secret_key = "replace-this-with-any-random-string"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"reply": "Say something and I'll respond!"})

    # Each visitor gets their own history stored in their browser's session
    # cookie, so two people chatting at once don't mix conversations.
    history = session.get("history", [])

    reply = get_bot_reply(message, history)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    session["history"] = history[-10:]  # keep last 5 turns

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)
