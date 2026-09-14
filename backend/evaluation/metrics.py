def pattern_score(repeated_count):
    return min(repeated_count, 5)

def relevance_score(is_code_quality):
    return 5 if is_code_quality else 2

def impact_score(failures):
    return min(failures, 5)

def specificity_score(actionable):
    return 5 if actionable else 2

def measurability_score(verifiable):
    return 5 if verifiable else 2