import json
from typing import Dict, Any, List


def export_compiled_to_json(discrete_bundle: Dict[str, Any], class_labels: List[str], path: str, clauses_num: int, L: int = 16, LF: int = 4) -> str:
    """
    Export compiled literals to a JSON compatible with Julia loader.
    discrete_bundle: dict with keys: positive, negative, positive_inv, negative_inv (each list[list[int]] 1-based)
    class_labels: list of class labels (strings); each class shares same clause sets for simplicity
    """
    data = {
        "clauses_num": int(clauses_num),
        "L": int(L),
        "LF": int(LF),
        "classes": []
    }
    for cls in class_labels:
        data["classes"].append({
            "class": str(cls),
            "positive": discrete_bundle["positive"],
            "negative": discrete_bundle["negative"],
            "positive_inv": discrete_bundle["positive_inv"],
            "negative_inv": discrete_bundle["negative_inv"]
        })
    if not path.endswith(".json"):
        path = path + ".json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path


