import nltk
import numpy as np
from difflib import get_close_matches
nltk.download('punkt_tab')
from nltk.stem.porter import PorterStemmer

stemmer=PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def correct_word(word, vocab, cutoff=0.75):
    """
    If 'word' isn't in the known vocabulary (all_words), try to find the
    closest known word (handles typos like 'helo' -> 'hello', 'thnks' -> 'thank').
    If nothing close enough is found, the original word is returned unchanged.
    """
    if word in vocab:
        return word

    matches = get_close_matches(word, vocab, n=1, cutoff=cutoff)
    return matches[0] if matches else word

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(i) for i in tokenized_sentence]

    # Try to auto-correct misspelled/near-miss words against the known vocabulary
    # before building the bag. This lets slightly wrong spellings still match.
    tokenized_sentence = [correct_word(w, all_words) for w in tokenized_sentence]

    bag=np.zeros(len(all_words),dtype=np.float32)

    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx]=1.0
    return bag