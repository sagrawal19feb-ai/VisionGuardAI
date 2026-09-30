"""Candidate gate protects existing classes when fine-tuning all four."""

import copy
import unittest

from visionguard.detection.model import LABEL_TO_ID
from visionguard.training.tune_all import selection_score


class SelectionTests(unittest.TestCase):
    def report(self):
        return {
            "classes": {
                label: {
                    "TP": 20,
                    "FP": 5,
                    "FN": 20,
                    "precision": 0.8,
                    "recall": 0.5,
                    "f1": 40 / 65,
                }
                for label in LABEL_TO_ID
            },
            "background_alarms": 0,
            "backgrounds": 20,
        }

    def test_improvement_requires_more_matched_knives_and_no_regressions(self):
        baseline = self.report()
        improved = copy.deepcopy(baseline)
        improved["classes"]["knife"].update(TP=23, FN=17, recall=23 / 40, f1=46 / 68)
        self.assertIsNotNone(selection_score(improved, baseline))
        weaker = copy.deepcopy(improved)
        weaker["classes"]["gun"].update(recall=0.3, f1=0.45)
        self.assertIsNone(selection_score(weaker, baseline))
        noisy = copy.deepcopy(improved)
        noisy["background_alarms"] = 3
        self.assertIsNone(selection_score(noisy, baseline))
        no_knife_gain = copy.deepcopy(improved)
        no_knife_gain["classes"]["knife"]["TP"] = 20
        self.assertIsNone(selection_score(no_knife_gain, baseline))


if __name__ == "__main__":
    unittest.main()
