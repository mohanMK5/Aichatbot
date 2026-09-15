"""
chat.py -- terminal version of the chatbot.

All the actual logic (classifier + confidence check + LLM fallback) lives
in bot_core.py now, so this file and app.py (the web version) both use
the exact same brain.
"""

from bot_core import get_bot_reply

print("Chatbot is ready! Type 'quit' to exit.")

# Simple in-memory history, only used to give the LLM fallback a bit of
# context. Not used by the classifier at all -- it stays stateless.
conversation_history = []

while True:

    sentence = input("You: ")

    if sentence.lower() == "quit":
        print("Bot: Goodbye! 👋")
        break

    reply = get_bot_reply(sentence, conversation_history)

    print("Bot:", reply)

    conversation_history.append({"role": "user", "content": sentence})
    conversation_history.append({"role": "assistant", "content": reply})
    conversation_history = conversation_history[-10:]
