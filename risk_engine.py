def calculate_risk(retrieved):
    """
    Transparent evidence-based screening score.

    This score is deliberately NOT presented as a statistical ML probability.
    It measures how many construction risk categories have meaningful semantic
    evidence in the uploaded project document.
    """
    thresholds = {
        "Progress & Schedule": 0.35,
        "Labor & Productivity": 0.35,
        "Materials & Procurement": 0.35,
        "Equipment": 0.35,
        "Design & Approvals": 0.35,
        "Weather": 0.40,
        "Communication": 0.35,
        "Cost & Commercial": 0.35,
    }

    risk_areas = 0
    for category, items in retrieved.items():
        if items:
            best_score = max(item["score"] for item in items)
            if best_score >= thresholds.get(category, 0.35):
                risk_areas += 1

    score = min(100, 10 + risk_areas * 11)
    level = "High" if score >= 65 else "Medium" if score >= 40 else "Low"

    return {
        "score": score,
        "level": level,
        "risk_areas": risk_areas,
    }
