# src/core/pareto_front.py
import logging
import numpy as np

logger = logging.getLogger(__name__)


def dominates(obj1_a, obj2_a, obj1_b, obj2_b):
    """
    Returns True if candidate 'a' strictly dominates candidate 'b'.
    (Assuming we want to MAXIMIZE both objectives).
    """
    return (obj1_a >= obj1_b and obj2_a >= obj2_b) and (obj1_a > obj1_b or obj2_a > obj2_b)


def calculate_crowding_distance(front_indices, obj1_scores, obj2_scores):
    """
    Calculates crowding distance for a specific Pareto front.
    Higher distance means the candidate is in a less crowded region (more diverse).
    """
    distances = {idx: 0.0 for idx in front_indices}

    if len(front_indices) <= 2:
        for idx in front_indices:
            distances[idx] = float('inf')
        return distances

    # Sort and calculate distance for Objective 1
    sorted_obj1 = sorted(front_indices, key=lambda i: obj1_scores[i])
    distances[sorted_obj1[0]] = float('inf')
    distances[sorted_obj1[-1]] = float('inf')
    obj1_range = obj1_scores[sorted_obj1[-1]] - obj1_scores[sorted_obj1[0]]
    if obj1_range > 0:
        for i in range(1, len(sorted_obj1) - 1):
            distances[sorted_obj1[i]] += (obj1_scores[sorted_obj1[i + 1]] - obj1_scores[
                sorted_obj1[i - 1]]) / obj1_range

    # Sort and calculate distance for Objective 2
    sorted_obj2 = sorted(front_indices, key=lambda i: obj2_scores[i])
    distances[sorted_obj2[0]] = float('inf')
    distances[sorted_obj2[-1]] = float('inf')
    obj2_range = obj2_scores[sorted_obj2[-1]] - obj2_scores[sorted_obj2[0]]
    if obj2_range > 0:
        for i in range(1, len(sorted_obj2) - 1):
            distances[sorted_obj2[i]] += (obj2_scores[sorted_obj2[i + 1]] - obj2_scores[
                sorted_obj2[i - 1]]) / obj2_range

    return distances


def select_top_k_pareto(candidates: list, k: int = 15) -> list:
    """
    Filters a list of candidates using NSGA-II style Non-Dominated Sorting.
    Requires 'predicted_affinity' and 'predicted_developability' keys.
    """
    if not candidates:
        return []

    # If untrained (no predictions), fallback to random selection to kickstart AF3
    if 'predicted_affinity' not in candidates[0]:
        logger.warning("No predictions found. Falling back to random selection.")
        np.random.shuffle(candidates)
        return candidates[:k]

    if len(candidates) <= k:
        return candidates

    obj1 = [c['predicted_affinity'] for c in candidates]
    obj2 = [c['predicted_developability'] for c in candidates]

    n = len(candidates)
    domination_counts = np.zeros(n, dtype=int)
    dominated_lists = [[] for _ in range(n)]
    fronts = [[]]

    # 1. Fast Non-Dominated Sort
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(obj1[i], obj2[i], obj1[j], obj2[j]):
                dominated_lists[i].append(j)
                domination_counts[j] += 1
            elif dominates(obj1[j], obj2[j], obj1[i], obj2[i]):
                dominated_lists[j].append(i)
                domination_counts[i] += 1

        if domination_counts[i] == 0:
            fronts[0].append(i)

    # Calculate subsequent fronts
    current_front = 0
    while current_front < len(fronts) and len(fronts[current_front]) > 0:  # <-- THE FIX
        next_front = []
        for i in fronts[current_front]:
            for j in dominated_lists[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        current_front += 1

    # 2. Select Top K using Fronts and Crowding Distance
    selected_indices = []

    for front in fronts:
        if len(selected_indices) + len(front) <= k:
            # We can take the whole front
            selected_indices.extend(front)
        else:
            # We need to take a subset of this front using Crowding Distance
            remaining_slots = k - len(selected_indices)
            distances = calculate_crowding_distance(front, obj1, obj2)

            # Sort front by crowding distance (descending)
            front_sorted = sorted(front, key=lambda idx: distances[idx], reverse=True)
            selected_indices.extend(front_sorted[:remaining_slots])
            break

    selected_candidates = [candidates[i] for i in selected_indices]

    # Log the summary of what we selected
    best_aff = max([c['predicted_affinity'] for c in selected_candidates])
    best_dev = max([c['predicted_developability'] for c in selected_candidates])
    logger.info(f"Pareto Selection complete. Extracted {len(selected_candidates)} candidates.")
    logger.info(f"Selected boundary metrics -> Max Affinity: {best_aff:.3f}, Max Developability: {best_dev:.3f}")

    return selected_candidates