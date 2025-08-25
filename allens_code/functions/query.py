from collections import deque
import heapq
import numpy as np
from .distance import sq_Euclidean_d

def ceil_radius_to_safe_level(r, safe_radii, eps=1e-10):
    """
    Return the smallest radius in safe_radii that is ≥ r,
    and slightly inflate it to prevent tight comparison errors later.
    """
    for radius in safe_radii:
        if radius >= r:
            return radius * (1 + eps)
    return float('inf')

def bfs_within_ball_dynamic(graph, dataset, root, k, query,
                            evaluated_set, traversable_radii_by_node,
                            percentile_radii_by_node=None):
    """
    Exact kNN search using dynamic radius and center-jumping strategy,
    with distance-prioritized neighbor exploration.

    Args:
        graph: dict[int, List[int]]
        dataset: np.ndarray of shape (n, d)
        root: int — starting index (seed)
        k: int — number of neighbors
        query: np.ndarray of shape (d,)
        evaluated_set: set[int] — tracks already evaluated points
        traversable_radii_by_node: dict[int, List[Tuple[int, float]]] — list of (index, radius)
        percentile_radii_by_node: dict[int, Dict[int, float]] or None — optional empirical radius per node

    Returns:
        visited: set[int]
        knn_list: list of (distance, index)
        jump_count: int
    """
    jump_count = 0
    visited = set()
    visited.add(root)
    expanded = set()

    # Priority queue stores (distance, node)
    queue = [(0.0, root)]

    knn_heap = []  # Max-heap of (-distance, index)
    best_idx = root
    best_dist = sq_Euclidean_d(query, dataset[root])
    evaluated_set.add(root)
    heapq.heappush(knn_heap, (-best_dist, root))

    jump_to_best = False

    current_empirical_radius = None
    if percentile_radii_by_node is not None:
        levels = percentile_radii_by_node[best_idx]
        sorted_items = sorted(levels.items())  # (index, radius)
        current_empirical_radius = float("inf")
        for _, r in sorted_items:
            if r >= best_dist:
                current_empirical_radius = r
                break

    while queue:
        if jump_to_best:
            current = best_idx
            jump_to_best = False
            jump_count += 1
            # Remove best_idx from queue if it’s in there (inefficient but rare)
            queue = [(d, n) for d, n in queue if n != best_idx]
            heapq.heapify(queue)
        else:
            _, current = heapq.heappop(queue)

        expanded.add(current)
        for neighbor in graph.get(current, []):
            if neighbor in visited:
                continue

            # Determine radius
            if percentile_radii_by_node is not None:
                radius = current_empirical_radius
            else:
                kth_dist = -knn_heap[0][0] if len(knn_heap) == k else float('inf')
                radius = (np.sqrt(best_dist) + np.sqrt(kth_dist))**2
                safe_radii = [r for _, r in traversable_radii_by_node[best_idx]]
                radius = ceil_radius_to_safe_level(radius, safe_radii)

            dist_to_best = sq_Euclidean_d(dataset[best_idx], dataset[neighbor])

            if dist_to_best <= radius:
                dist_to_q = sq_Euclidean_d(query, dataset[neighbor])
                visited.add(neighbor)
                heapq.heappush(queue, (dist_to_q, neighbor))

                if neighbor not in evaluated_set:
                    evaluated_set.add(neighbor)
                    if len(knn_heap) < k:
                        heapq.heappush(knn_heap, (-dist_to_q, neighbor))
                    elif dist_to_q < -knn_heap[0][0]:
                        heapq.heappushpop(knn_heap, (-dist_to_q, neighbor))

                    if dist_to_q < best_dist:
                        best_dist = dist_to_q
                        best_idx = neighbor
                        jump_to_best = True

                        # Update empirical radius if applicable
                        if percentile_radii_by_node is not None:
                            levels = percentile_radii_by_node[best_idx]
                            sorted_items = sorted(levels.items())  # (index, radius)
                            current_empirical_radius = float("inf")
                            for _, r in sorted_items:
                                if r >= best_dist:
                                    current_empirical_radius = r
                                    break

    knn_list = sorted([(-d, i) for d, i in knn_heap], key=lambda x: x[0])
    return expanded, knn_list, jump_count

def query_phase(query, index, dataset, graph, safe_radii_by_node, percentile_radii_by_node=None, k=5):
    """
    Runs a query via FAISS HNSW (coarse) and exact BFS (fine).

    Args:
        query: np.ndarray of shape (d,)
        index: FAISS index
        dataset: np.ndarray
        graph: dict[int, List[int]]
        safe_radii_by_node: dict[int, List[(int,float)]]
        k: int

    Returns:
        visited: set[int]
        knn_list: list of (distance, index)
        jump_count: int
    """
    # Initial HNSW search
    D, I = index.search(query.reshape(1, -1), k)
    root = I[0][0]

    visited, knn_list, jump_count = bfs_within_ball_dynamic(
        graph=graph,
        dataset=dataset,
        root=root,
        k=k,
        query=query,
        evaluated_set=set(),
        traversable_radii_by_node=safe_radii_by_node,
        percentile_radii_by_node=percentile_radii_by_node
    )

    return visited, knn_list, jump_count