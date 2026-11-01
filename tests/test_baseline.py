import unittest

from lib.data import development
from lib.evaluation import auc, metrics, select_threshold
from lib.model import OOS, Router, features, reject, tokens


class RoutingTests(unittest.TestCase):
    def test_tokens_preserve_negation_and_stem(self):
        self.assertEqual(tokens("NOT running, don't cancel!"), ["not", "run", "don't", "cancel"])

    def test_features_do_not_join_punctuation_delimited_sentences(self):
        expected = {("red",), ("blue",), ("green",), ("red", "blue")}
        for boundary in (". ", "! ", "? ", "?! ", "\n"):
            with self.subTest(boundary=boundary):
                self.assertEqual(features("red blue" + boundary + "green", False, 3), expected)

    def test_feature_sets_and_decimal_token_policy(self):
        self.assertEqual(features("3.50", False), {("3",), ("50",), ("3", "50")})
        self.assertEqual(features("pay pay", False), features("pay pay pay", False))
        self.assertEqual(features("...?!\n", False), set())

    def test_match_score_and_deterministic_tie(self):
        router = Router([["red blue", "a"], ["red blue", "b"]], stemming=False, ngram_max=1)
        self.assertEqual(router.match("red green"), ("a", 1 / 3, 0))
        self.assertEqual(router.match("unknown"), (OOS, 0.0, None))
        self.assertEqual(reject("a", 1 / 3, .4), OOS)
        self.assertEqual(reject("a", .4, .4), "a")
        self.assertEqual(reject("a", 0, 0), OOS)

    def test_training_rejects_oos(self):
        with self.assertRaises(ValueError):
            Router([["unknown", OOS]])

    def test_auc_ties(self):
        self.assertEqual(auc(["a", OOS], [.1, .9]), 1)
        self.assertEqual(auc(["a", OOS], [.5, .5]), .5)
        self.assertEqual(auc(["a", OOS], [.9, .1]), 0)

    def test_metrics_rejections_count_as_in_scope_errors(self):
        result = metrics(["a", "b", OOS], ["a", OOS, OOS], ["a", "b"], [0, .8, .9])
        self.assertEqual(result["in_scope_macro_f1"], .5)
        self.assertEqual(result["in_scope_rejection_rate"], .5)
        self.assertAlmostEqual(result["oos_f1"], 2 / 3)

    def test_threshold_selection(self):
        best, _ = select_threshold(["a", OOS], [("a", .8, 0), ("a", .2, 0)], ["a"], [0, .5, 1])
        self.assertEqual(best["threshold"], .5)
        self.assertEqual(best["selection_score"], 1)

    def test_development_never_requires_test_or_oos_training(self):
        data = {
            "train": [["training", str(i)] for i in range(150) for _ in range(100)],
            "val": [["validation", str(i)] for i in range(150) for _ in range(20)],
            "oos_val": [["unknown", "oos"] for _ in range(100)],
        }
        train, validation, labels = development(data)
        self.assertEqual((len(train), len(validation), len(labels)), (15000, 3100, 150))
        self.assertEqual(validation[-1], ["unknown", OOS])


if __name__ == "__main__":
    unittest.main()
