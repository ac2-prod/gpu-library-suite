"""Deterministic implementation and problem-size assignment rules."""

from typing import Any, Dict, List, Mapping, Sequence, Tuple


IMPLEMENTATION_PERMUTATIONS = (
    ("cpu", "cuda", "openacc"),
    ("cpu", "openacc", "cuda"),
    ("cuda", "cpu", "openacc"),
    ("cuda", "openacc", "cpu"),
    ("openacc", "cpu", "cuda"),
    ("openacc", "cuda", "cpu"),
)


def permutation_index(node_index: int, wave: int) -> int:
    if node_index < 0 or wave < 0:
        raise ValueError("node index and wave must be non-negative")
    return (node_index + wave) % len(IMPLEMENTATION_PERMUTATIONS)


def implementation_order(node_index: int, wave: int) -> Tuple[str, str, str]:
    return IMPLEMENTATION_PERMUTATIONS[permutation_index(node_index, wave)]


def size_order_index(node_index: int) -> int:
    if node_index < 0:
        raise ValueError("node index must be non-negative")
    return node_index % 2


def ordered_cases(cases: Sequence[Mapping[str, Any]], node_index: int) -> List[Mapping[str, Any]]:
    result = list(cases)
    if size_order_index(node_index) == 1:
        result.reverse()
    return result


def assignment_counts(node_count: int, wave: int) -> Dict[str, Dict[str, int]]:
    if node_count <= 0:
        raise ValueError("node_count must be positive")
    permutations = {str(index): 0 for index in range(6)}
    size_orders = {str(index): 0 for index in range(2)}
    for node_index in range(node_count):
        permutations[str(permutation_index(node_index, wave))] += 1
        size_orders[str(size_order_index(node_index))] += 1
    return {
        "permutation_assignment_counts": permutations,
        "size_order_assignment_counts": size_orders,
    }


def multiwave_assignment_counts(node_count: int, waves: Sequence[int]) -> Dict[str, Dict[str, int]]:
    permutations = {str(index): 0 for index in range(6)}
    size_orders = {str(index): 0 for index in range(2)}
    joint = {
        "{0}:{1}".format(permutation, size_order): 0
        for permutation in range(6)
        for size_order in range(2)
    }
    for wave in waves:
        for node_index in range(node_count):
            permutation = permutation_index(node_index, wave)
            size_order = size_order_index(node_index)
            permutations[str(permutation)] += 1
            size_orders[str(size_order)] += 1
            joint["{0}:{1}".format(permutation, size_order)] += 1
    return {
        "permutation_assignment_counts": permutations,
        "size_order_assignment_counts": size_orders,
        "permutation_size_order_counts": joint,
    }
