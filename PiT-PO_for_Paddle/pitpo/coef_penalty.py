import re
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def count_param_occurrences_in_text(text: str, indices: Iterable[int]) -> Dict[int, int]:
    """Count occurrences of params[i] in the given text."""
    counts: Dict[int, int] = {}
    if not text:
        return counts
    for i in indices:
        pattern = rf"params\s*\[\s*{i}\s*\]"
        matches = re.findall(pattern, text)
        counts[i] = len(matches)
    return counts


def _aggregate_params(
    optimized_params_by_test: Dict[str, Sequence[float]],
) -> Dict[int, float]:
    """Aggregate absolute coefficient values across tests (take mean |b_i|)."""
    sums: Dict[int, float] = {}
    cnts: Dict[int, int] = {}
    for _k, params in (optimized_params_by_test or {}).items():
        for i, val in enumerate(params or []):
            try:
                v = abs(float(val))
            except Exception:
                continue
            sums[i] = sums.get(i, 0.0) + v
            cnts[i] = cnts.get(i, 0) + 1
    mean_abs: Dict[int, float] = {}
    for i in sums:
        mean_abs[i] = sums[i] / max(cnts[i], 1)
    return mean_abs


def find_redundant_indices(
    mean_abs: Dict[int, float],
    ratio_threshold: float = 0.05,
    eps: float = 1e-30,
) -> Tuple[List[int], Dict[int, float]]:
    """Identify redundant term indices via the relative ratio tau_i (Paper Eq. 5).

    tau_i = |b_i| / (sum_j |b_j| + eps)
    Redundant if tau_i <= ratio_threshold (rho).

    Returns (redundant_indices, tau_map).
    """
    total = sum(mean_abs.values()) + eps
    tau_map: Dict[int, float] = {i: v / total for i, v in mean_abs.items()}
    redundant = sorted([i for i, tau in tau_map.items() if tau <= ratio_threshold])
    return redundant, tau_map


def compute_coefficient_penalty(
    response_text: str,
    optimized_params_by_test: Optional[Dict[str, Sequence[float]]] = None,
    *,
    ratio_threshold: float = 0.05,
    penalty_weight: float = 1.0,
    eps: float = 1e-30,
    # Keep old keyword for backward compatibility (ignored if ratio_threshold given)
    threshold: Optional[float] = None,
) -> Tuple[float, Dict[str, object]]:
    """Compute redundancy penalty based on Support Exclusion Theorem (Paper Eq. 5-6).

    For each redundant term i (tau_i <= rho) that appears in response_text:
        penalty_i = p * max(0, -ln(|b_i| + eps))   (Eq. 6, natural log)
    Total penalty = sum over redundant indices.

    Returns (total_penalty, details_dict).
    """
    empty = {"redundant_indices": [], "tau_map": {}, "occurrence_counts": {},
             "penalty_by_index": {}, "mean_abs": {}}
    if not response_text:
        return 0.0, empty

    mean_abs = _aggregate_params(optimized_params_by_test or {})
    if not mean_abs:
        return 0.0, empty

    redundant_indices, tau_map = find_redundant_indices(mean_abs, ratio_threshold, eps)
    if not redundant_indices:
        return 0.0, {**empty, "tau_map": tau_map, "mean_abs": mean_abs}

    counts = count_param_occurrences_in_text(response_text, redundant_indices)
    used = [i for i in redundant_indices if counts.get(i, 0) > 0]

    penalty_by_index: Dict[int, float] = {}
    for i in used:
        v = max(mean_abs.get(i, 0.0), eps)
        penalty_by_index[i] = float(penalty_weight) * float(max(-math.log(v + eps), 0.0))

    total_penalty = float(sum(penalty_by_index.values()))

    return total_penalty, {
        "redundant_indices": redundant_indices,
        "tau_map": tau_map,
        "occurrence_counts": counts,
        "total_occurrences": int(sum(counts.values())),
        "penalty_by_index": penalty_by_index,
        "mean_abs": mean_abs,
        "used_indices": used,
    }


# Keep old names for backward compatibility
def aggregate_small_params(
    optimized_params_by_test: Dict[str, Sequence[float]],
    threshold: float,
) -> Tuple[List[int], Dict[int, float]]:
    """Backward-compatible wrapper: now uses ratio-based detection."""
    mean_abs = _aggregate_params(optimized_params_by_test)
    redundant, _ = find_redundant_indices(mean_abs, ratio_threshold=0.05)
    min_vals = {i: mean_abs.get(i, 0.0) for i in redundant}
    return redundant, min_vals
