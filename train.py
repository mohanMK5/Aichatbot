import json

from nlp import tokenize, stem, bag_of_words

import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)


all_words = []
tags = []
xy = []


for intent in intents['intents']:

    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:

        w = tokenize(pattern)
        all_words.extend(w)

        xy.append((w, tag))


ignore = ['&', '?', '!', ',', '.']


all_words = [
    stem(i)
    for i in all_words
    if i not in ignore
]

all_words = sorted(set(all_words))
tags = sorted(set(tags))


X_data = []
y_data = []


for (sentence, tag) in xy:

    bag = bag_of_words(sentence, all_words)

    X_data.append(bag)
    y_data.append(tags.index(tag))


X_data = np.array(X_data)
y_data = np.array(y_data)


X_train, X_test, y_train, y_test = train_test_split(
    X_data,
    y_data,
    test_size=0.2,
    random_state=42,
    stratify=y_data
)


model = LogisticRegression(max_iter=1000, C=10)

model.fit(X_train, y_train)


predictions = model.predict(X_test)

score = model.score(X_test, y_test)

print(f"Test accuracy: {score:.2f}")
print(classification_report(
    y_test,
    predictions,
    labels=list(range(len(tags))),
    target_names=tags,
    zero_division=0
))

# The split above is only for evaluating how well the model generalizes.
# For the model we actually ship to the chatbot, refit on ALL the data
# (train_test_split throws away 20% of an already tiny dataset otherwise).
model.fit(X_data, y_data)



#while True:
#    sentence = input("You: ")

#    if sentence.lower() == "quit":
#        break

 #   words = tokenize(sentence)

 #   bag = bag_of_words(words, all_words)

 #   bag = np.array(bag)

 #   prediction = model.predict([bag])[0]

 #   intent = tags[prediction]

 #   print("Predicted intent:", intent)

    

joblib.dump(model, 'chatbot_model.pkl')
joblib.dump(all_words, 'all_words.pkl')
joblib.dump(tags, 'tags.pkl')

print("Model saved successfully!")