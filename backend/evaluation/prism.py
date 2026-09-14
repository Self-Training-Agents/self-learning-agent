from .metrics import (
    pattern_score,
    relevance_score,
    impact_score,
    specificity_score,
    measurability_score,
)

class PrismEvaluator:
    """PRISM scoring engine."""

    THRESHOLD = 18

    def evaluate(self, candidate):
        scores = {
            "pattern": pattern_score(candidate["repeated"]),
            "relevance": relevance_score(candidate["code_quality"]),
            "impact": impact_score(candidate["failures"]),
            "specificity": specificity_score(candidate["actionable"]),
            "measurability": measurability_score(candidate["verifiable"]),
        }

        total_score = sum(scores.values())

        return {
            "learning": candidate["learning"],
            "scores": scores,
            "total_score": total_score,
            "decision": "SAVE" if total_score >= self.THRESHOLD else "REJECT",
        }