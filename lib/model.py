"""Reusable lexical router: regex tokens, Porter stems, NLTK n-grams, Jaccard."""

import re
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer
from nltk.util import ngrams

OOS = "out_of_scope"
STEMMER = PorterStemmer()


def tokens(text, stemming=True):
    words = re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text.lower().replace("’", "'"))
    return [STEMMER.stem(word) for word in words] if stemming else words


def features(text, stemming=True, ngram_max=2):
    # A boundary sketch, not linguistic sentence segmentation: abbreviations
    # such as "Dr. Ali" split too; "card.What" has no recognized boundary.
    # Split BEFORE discarding punctuation, keeping decimals such as 3.50 intact.
    chunks = re.split(r"[.!?]+(?=\s|$)|[\r\n]+", text)
    grams = set()
    for chunk in chunks:
        words = tokens(chunk, stemming)
        for n in range(1, ngram_max + 1):
            grams.update(ngrams(words, n))
    return grams


class Router:
    def __init__(self, train, stemming=True, ngram_max=2):
        if not train or any(label == OOS for _, label in train):
            raise ValueError("Fit requires nonempty in-scope training data only")
        if ngram_max not in (1, 2, 3):
            raise ValueError("ngram_max must be 1, 2, or 3")
        self.train, self.stemming, self.ngram_max = train, stemming, ngram_max
        self.sizes = []
        self.postings = defaultdict(list)
        for index, (text, _) in enumerate(train):
            grams = features(text, stemming, ngram_max)
            self.sizes.append(len(grams))
            for gram in grams:
                self.postings[gram].append(index)

    def match(self, text):
        grams = features(text, self.stemming, self.ngram_max)
        # Count overlaps only for training examples sharing a feature.
        overlaps = Counter()
        for gram in sorted(grams):
            overlaps.update(self.postings.get(gram, ()))
        best_index, best_score = None, 0.0
        for index in sorted(overlaps):
            score = overlaps[index] / (len(grams) + self.sizes[index] - overlaps[index])
            if score > best_score:  # Ties keep the first training example.
                best_index, best_score = index, score
        label = self.train[best_index][1] if best_index is not None else OOS
        return label, best_score, best_index


def reject(label, score, threshold):
    return OOS if score == 0 or score < threshold else label
