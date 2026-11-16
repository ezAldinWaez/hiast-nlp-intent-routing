import unittest

from lib.linguistic import LinguisticRouter, representation
from lib.model import OOS


def word(index, text, tag, head, relation, lemma=None):
    return dict(id=index, text=text, lemma=lemma or text.lower(), upos=tag,
                head=head, deprel=relation)


class LinguisticTests(unittest.TestCase):
    def test_arguments_case_negation_and_function_words(self):
        record = [[
            word(1, "Do", "AUX", 3, "aux", "do"),
            word(2, "n't", "PART", 3, "advmod", "not"),
            word(3, "transfer", "VERB", 0, "root"),
            word(4, "money", "NOUN", 3, "obj"),
            word(5, "from", "ADP", 7, "case"),
            word(6, "my", "PRON", 7, "nmod:poss"),
            word(7, "accounts", "NOUN", 3, "obl", "account"),
            word(8, ".", "PUNCT", 3, "punct"),
        ]]
        lexical, structural = representation(record)
        self.assertEqual(lexical, {"do", "not", "transfer", "money", "from", "my", "account"})
        self.assertEqual(structural, {
            ("transfer", "obj", "money"), ("transfer", "obl", "account"),
            ("account", "case", "from"), ("transfer", "neg", "not"),
        })
        record[0][4]["lemma"] = "to"
        self.assertIn(("account", "case", "to"), representation(record)[1])
        self.assertNotIn(("account", "case", "from"), representation(record)[1])

    def test_subtypes_and_sentence_local_heads(self):
        records = [
            [word(1, "money", "NOUN", 2, "nsubj:pass"),
             word(2, "transferred", "VERB", 0, "root", "transfer")],
            [word(1, "rain", "NOUN", 0, "root")],
        ]
        self.assertEqual(representation(records)[1], {("transfer", "nsubj:pass", "money")})
        self.assertEqual(representation([]), (set(), set()))

    def test_weighted_similarity_and_each_fallback_direction(self):
        training = [["first", "a"]]
        # lexical Jaccard = 1/3, structural Jaccard = 1/2
        router = LinguisticRouter(training, [({"a", "b"}, {("v", "obj", "n")})], .25)
        query = ({"a", "c"}, {("v", "obj", "n"), ("w", "obj", "m")})
        self.assertAlmostEqual(router.match(query)[1], .75 / 3 + .25 / 2)
        self.assertAlmostEqual(router.match((query[0], set()))[1], 1 / 3)
        empty_train = LinguisticRouter(training, [({"a", "b"}, set())], .5)
        self.assertAlmostEqual(empty_train.match(query)[1], 1 / 3)

    def test_structural_match_can_find_a_lexically_disjoint_neighbor(self):
        router = LinguisticRouter(
            [["a", "first"], ["b", "second"]],
            [({"x"}, {("v", "obj", "a")}), ({"y"}, {("v", "obj", "b")})], .5,
        )
        self.assertEqual(router.match(({"z"}, {("v", "obj", "b")})), ("second", .5, 1))

    def test_zero_weight_ties_empty_query_and_no_overlap(self):
        train = [["a", "first"], ["b", "second"]]
        reps = [({"x"}, set()), ({"x"}, {("v", "obj", "n")})]
        router = LinguisticRouter(train, reps, 0)
        self.assertEqual(router.match(({"x"}, {("v", "obj", "n")})), ("first", 1, 0))
        self.assertEqual(router.match((set(), set())), (OOS, 0, None))
        self.assertEqual(router.match(({"unknown"}, set())), (OOS, 0, None))

    def test_invalid_fit(self):
        for train, reps, weight in [([], [], 0), ([["x", OOS]], [(set(), set())], 0),
                                    ([["x", "a"]], [], 0),
                                    ([["x", "a"]], [(set(), set())], -1)]:
            with self.subTest(train=train, weight=weight), self.assertRaises(ValueError):
                LinguisticRouter(train, reps, weight)


if __name__ == "__main__":
    unittest.main()
