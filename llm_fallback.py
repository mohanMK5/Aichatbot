"""
llm_fallback.py

This is the "second brain" of the chatbot.

The classifier in chat.py (bag-of-words + LogisticRegression) is fast and
predictable, but it can only recognize the exact intents it was trained on
in intents.json. Anything outside that (casual small talk it's never seen,
slightly unusual phrasing, etc.) gets low confidence.

Instead of always showing a generic "I don't understand" message in that
case, we hand the message over to a real LLM just for that one reply, so
the conversation still feels natural.

Using Groq here because it has a genuinely free tier (no credit card, just
rate limits), which matters for testing and for a future fullstack deploy
where you don't want a per-message bill.

Design choices, on purpose, for simplicity:
- Only ONE function is exposed: get_llm_reply(). Everything else is
  internal to this file.
- If no API key is configured, or the 'groq' package isn't installed,
  this quietly returns None instead of crashing. chat.py checks for None
  and uses its own canned fallback message in that case. The classifier
  part of the project keeps working with zero setup either way.
"""

import os

# Guard the import: if the 'groq' package isn't installed, the rest of
# the chatbot (the classifier) should still run fine without it.
try:
    from groq import Groq
    _sdk_available = True
except ImportError:
    _sdk_available = False

_api_key = os.environ.get("GROQ_API_KEY")
_client = Groq(api_key=_api_key) if (_sdk_available and _api_key) else None

# Small, fast, free-tier model. Good enough for short casual replies.
# Groq renames/retires model ids periodically -- if this ever 404s again,
# run the snippet at the bottom of this file to list currently active ids.
_MODEL = "openai/gpt-oss-20b"

# Keep the LLM's role narrow on purpose: casual chit-chat only, not a
# general-purpose assistant. This keeps its behavior consistent with the
# "fun chat buddy" personality defined in intents.json.
_SYSTEM_PROMPT = (
    "You are a friendly, casual chatbot having light small talk with the user. "
    "Keep every reply short, 1-2 sentences, warm and conversational, similar in tone "
    "to the sample responses you'd see in a casual chatbot's intents file. "
    "Do not answer technical, coding, medical, legal, or research questions in depth. "
    "If asked something like that, gently redirect back to casual conversation."
)

# How many past turns (user+bot pairs) to remember for context.
# Kept small on purpose: this is a chit-chat fallback, not a full assistant.
_MAX_HISTORY_TURNS = 5


def get_llm_reply(user_message, history=None):
    """
    Returns a short conversational reply from the LLM, or None if it
    isn't available (no package installed / no API key / a request error).

    'history' is an optional list of {"role": "user"/"assistant", "content": str}
    dicts, oldest first, used only to give the LLM a little conversational context.
    """
    if _client is None:
        return None

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-_MAX_HISTORY_TURNS * 2:])
    messages.append({"role": "user", "content": user_message})

    try:
        response = _client.chat.completions.create(
            model=_MODEL,
            max_tokens=150,
            messages=messages,
        )
        return response.choices[0].message.content.strip()
    except Exception as error:
        # Any API/network issue: fail quietly, let chat.py use its canned fallback.
        print(f"[llm_fallback] Could not reach the LLM ({error}); using canned fallback.")
        return None


# If _MODEL above ever comes back as "does not exist" again, Groq has
# retired that id. Run this file directly (python llm_fallback.py) to
# print every model id your key currently has access to, then update
# _MODEL above to one of them.
if __name__ == "__main__":
    if _client is None:
        print("Set GROQ_API_KEY first, then re-run this file.")
    else:
        for m in _client.models.list().data:
            print(m.id)