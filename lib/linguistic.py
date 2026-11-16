"""M2: predicted linguistic annotations and two-channel nearest-example routing."""

from collections import Counter, defaultdict

from .model import OOS

# Same English no-character-LM packages as the S04 tools appendix; no NER.
PACKAGES = {
    "tokenize": "combined_nocharlm",
    "mwt": "combined",
    "pos": "combined_nocharlm",
    "lemma": "combined_nocharlm",
    "depparse": "combined_nocharlm",
}
NEGATION = {"not", "n't", "never", "no"}
ARGUMENTS = {"nsubj", "csubj", "obj", "iobj", "obl"}


def english_pipeline(model_dir):
    """Load explicitly installed S04-style English models on CPU, without downloads."""
    import stanza

    return stanza.Pipeline(
        lang="en", dir=str(model_dir), package=None, processors=PACKAGES,
        use_gpu=False, download_method=None, verbose=False,
    )


def annotate(texts, pipeline, batch_size=128):
    """Return sentence-local word records, preserving request order and word IDs."""
    records = []
    for start in range(0, len(texts), batch_size):
        docs = pipeline.bulk_process(texts[start:start + batch_size])
        records.extend([
            [
                [dict(id=w.id, text=w.text, lemma=w.lemma, upos=w.upos,
                      head=w.head, deprel=w.deprel) for w in sentence.words]
                for sentence in doc.sentences
            ]
            for doc in docs
        ])
    if len(records) != len(texts):
        raise ValueError("Annotation count does not match request count")
    return records


def lemma(word):
    return (word["lemma"] or word["text"]).lower().replace("’", "'")


def representation(sentences):
    """Lemma unigrams and argument triples, with case markers and negation cues.

    An argument triple retains its exact relation, e.g. nsubj:pass. Case and
    negation are additional triples: (argument, case, from), (verb, neg, not).
    These are syntactic cues, not semantic roles or a negation-scope analysis.
    """
    lexical, structural = set(), set()
    for words in sentences:
        by_id = {word["id"]: word for word in words}
        lexical.update(lemma(w) for w in words if w["upos"] not in {"PUNCT", "SYM"})
        for word in words:
            head = by_id.get(word["head"])
            if head is None:
                continue
            relation = word["deprel"]
            if relation.split(":")[0] in ARGUMENTS:
                structural.add((lemma(head), relation, lemma(word)))
                # In UD, a preposition normally attaches to the argument noun.
                for child in words:
                    if child["head"] == word["id"] and child["deprel"] == "case":
                        structural.add((lemma(word), "case", lemma(child)))
            if lemma(word) in NEGATION:
                structural.add((lemma(head), "neg", lemma(word)))
    return lexical, structural


class LinguisticRouter:
    """Nearest example under lexical/structural Jaccard, with lexical fallback."""

    def __init__(self, train, representations, weight=0):
        if not train or any(label == OOS for _, label in train):
            raise ValueError("Fit requires nonempty in-scope training data only")
        if len(train) != len(representations):
            raise ValueError("One representation is required per training request")
        if not 0 <= weight <= 1:
            raise ValueError("weight must be in [0, 1]")
        self.train, self.weight = train, weight
        self.sizes = [[], []]
        self.postings = [defaultdict(list), defaultdict(list)]
        for index, channels in enumerate(representations):
            for channel, grams in enumerate(channels):
                self.sizes[channel].append(len(grams))
                for gram in grams:
                    self.postings[channel][gram].append(index)

    def match(self, query):
        overlaps = [Counter(), Counter()]
        for channel, grams in enumerate(query):
            if channel == 1 and self.weight == 0:
                continue
            for gram in sorted(grams):
                overlaps[channel].update(self.postings[channel].get(gram, ()))
        best_index, best_score = None, 0.0
        # Any example with a positive score must share at least one feature.
        for index in sorted(overlaps[0].keys() | overlaps[1].keys()):
            scores = []
            for channel in (0, 1):
                shared = overlaps[channel][index]
                union = len(query[channel]) + self.sizes[channel][index] - shared
                scores.append(shared / union if union else 0.0)
            weight = self.weight if query[1] and self.sizes[1][index] else 0
            score = (1 - weight) * scores[0] + weight * scores[1]
            if score > best_score:
                best_index, best_score = index, score
        label = self.train[best_index][1] if best_index is not None else OOS
        return label, best_score, best_index
