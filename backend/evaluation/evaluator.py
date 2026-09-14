from .prism import PrismEvaluator

candidate = {
    "learning": "Run tests before editing",
    "repeated": 4,
    "code_quality": True,
    "failures": 3,
    "actionable": True,
    "verifiable": True
}

result = PrismEvaluator().evaluate(candidate)

print("\nPRISM Evaluation")
print("----------------")
print("Learning :", result["learning"])
print("Scores   :", result["scores"])
print("Total    :", result["total_score"])
print("Decision :", result["decision"])