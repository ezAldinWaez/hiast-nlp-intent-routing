import unittest

from model import tokens, OOS, Router, reject
from evaluation import auc, metrics, select_threshold
from dataset import development


class RoutingTests(unittest.TestCase):
    def test_tokens_preserve_negation_and_stem(self):
        self.assertEqual(tokens("NOT running, don't cancel!"), ["not", "run", "don't", "cancel"])

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
