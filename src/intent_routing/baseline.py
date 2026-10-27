"""Regex tokens, optional Porter stemming, and nearest-example n-gram overlap."""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from nltk.stem import PorterStemmer

OOS = "out_of_scope"
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+")
NEGATIONS = {"no", "not", "never", "cannot", "without"}


@dataclass(frozen=True)
class Features:
    ngram_max: int = 2
    stemming: bool = True
    drop_negation: bool = False

    def __post_init__(self):
        if self.ngram_max not in (1, 2, 3):
            raise ValueError("ngram_max must be 1, 2, or 3")

    def tokens(self, text):
        tokens = TOKEN.findall(text.lower().replace("\u2019", "'"))
        if self.drop_negation:
            tokens = [t for t in tokens if t not in NEGATIONS and not t.endswith("n't")]
        if self.stemming:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(t) for t in tokens]
        return tokens

    def __call__(self, text):
        tokens = self.tokens(text)
        return {tuple(tokens[i:i + n]) for n in range(1, self.ngram_max + 1)
                for i in range(len(tokens) - n + 1)}


class Router:
    """Maximum Jaccard overlap against training examples; ties use source order.

    The score is a similarity, not a calibrated probability. An inverted index
    avoids scoring training examples with no shared features.
    """

    def __init__(self, train, features):
        if not train or any(label == OOS for _, label in train):
            raise ValueError("Fit requires nonempty in-scope training data only")
        self.features = features
        self.train = train
        self.sizes = []
        self.postings = defaultdict(list)
        for index, (text, _) in enumerate(train):
            grams = features(text)
            self.sizes.append(len(grams))
            for gram in grams:
                self.postings[gram].append(index)

    def match(self, text):
        grams = self.features(text)
        intersections = Counter()
        for gram in sorted(grams):
            intersections.update(self.postings.get(gram, ()))
        if not intersections:
            return OOS, 0.0, None
        best = min(intersections, key=lambda i: (
            -intersections[i] / (len(grams) + self.sizes[i] - intersections[i]), i))
        score = intersections[best] / (len(grams) + self.sizes[best] - intersections[best])
        return self.train[best][1], score, best


def reject(label, score, threshold):
    return OOS if score == 0 or score < threshold else label
