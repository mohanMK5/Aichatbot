"""
bot_core.py

The actual "brain" of the chatbot, pulled out on its own so both the
terminal version (chat.py) and the web version (app.py) can call the
exact same logic. Nothing about the classifier concept changes here --
this is the same bag-of-words + LogisticRegression + confidence-threshold
pipeline as before, just wrapped in one reusable function.
"""

import json
import random
import joblib
import numpy as np

from nlp import tokenize, bag_of_words
from llm_fallback import get_llm_reply

# Load the saved model once, when this module is first imported.
model = joblib.load('chatbot_model.pkl')
all_words = joblib.load('all_words.pkl')
tags = joblib.load('tags.pkl')

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# If the model isn't at least this confident about the best-matching tag,
# it's safer to admit we're unsure than to force an irrelevant response.
CONFIDENCE_THRESHOLD = 0.55

FALLBACK_RESPONSES = [
    "I'm not quite sure I understood that. Could you rephrase it?",
    "Hmm, I didn't catch that clearly. Can you say it another way?",
    "I'm not confident I understood — could you try different words?",
]


def get_bot_reply(sentence, history=None):
    """
    Runs one message through the full pipeline and returns the bot's
    reply as a plain string.

    'history' is an optional list of {"role": ..., "content": ...} dicts,
    passed straight through to the LLM fallback for conversational
    context. It is NOT used by the classifier itself.
    """
    words = tokenize(sentence)
    bag = np.array(bag_of_words(words, all_words))

    probabilities = model.predict_proba([bag])[0]
    best_idx = np.argmax(probabilities)
    confidence = probabilities[best_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        llm_reply = get_llm_reply(sentence, history)
        if llm_reply:
            return llm_reply
        return random.choice(FALLBACK_RESPONSES)

    tag = tags[best_idx]
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    # Safety net: should never happen since 'tag' always comes from 'tags',
    # but keeps the function from ever returning nothing.
    return random.choice(FALLBACK_RESPONSES)
