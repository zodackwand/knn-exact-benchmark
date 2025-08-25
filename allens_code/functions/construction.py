import numpy as np
from collections import deque, defaultdict
from .distance import sq_Euclidean_d

def bfs_within_ball(graph, dataset, root, radius):
    """Returns all nodes reachable from `root` within `radius` using outgoing edges."""
    visited = set()
    queue = deque([root])
    visited.add(root)
    center = dataset[root]

    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor in visited:
                continue
            if sq_Euclidean_d(dataset[neighbor],center) <= radius:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited

def build_ball_traversable_graph(dataset):
    """
    Construct a directed graph such that for every node v and each radius corresponding
    to the 1, 2, 4, 8, ..., max_neighbors-th closest neighbors (including the furthest one),
    the subgraph induced by B_r(v) is traversable from v.

    Args:
        dataset: np.ndarray of shape (n, d)
        max_neighbors: int — maximum number of neighbors to consider (default: all)

    Returns:
        graph: dict[int, List[int]] — directed adjacency list
        straversable_radii_by_node: dict[int, List[float]] — guaranteed traversable radii per node
    """
    n = len(dataset)
    graph = defaultdict(list)
    traversable_radii_by_node = {}

    for v in range(n):
        # Compute and sort distances from v to all other nodes
        dists = [(u, sq_Euclidean_d(dataset[v],dataset[u])) for u in range(n) if u != v]
        dists.sort(key=lambda x: x[1])

        # Select geometric progression indices
        traversable_indices = [0]
        i = 1
        while i <= len(dists):
            traversable_indices.append(i)
            i *= 2

        # Ensure the final index is included
        if len(dists) not in traversable_indices:
            traversable_indices.append(len(dists))

        # Process each safe radius
        traversable_radii = []
        for idx in traversable_indices:
            u_subset = dists[:idx + 1]
            max_r = u_subset[-1][1]
            traversable_radii.append((idx, max_r))

            reachable = bfs_within_ball(graph, dataset, v, max_r)

            for u, dist in u_subset:
                if u in reachable:
                    continue
                if not reachable:
                    graph[v].append(u)
                else:
                    closest = min(
                        reachable,
                        key=lambda w: sq_Euclidean_d(dataset[u],dataset[w])
                    )
                    graph[closest].append(u)

                # Update reachability after each new edge
                reachable = bfs_within_ball(graph, dataset, v, max_r)

        traversable_radii_by_node[v] = traversable_radii

    return graph, traversable_radii_by_node

def compute_exact_knns(dataset, k):
    """
    Compute the exact kNNs for each point in the dataset using brute-force search.

    Args:
        dataset: np.ndarray of shape (n, d)
        k: int — number of nearest neighbors

    Returns:
        knns_by_node: dict[int, List[Tuple[int, float]]]
                      Maps each node index to list of (neighbor index, distance)
    """
    n = len(dataset)
    knns_by_node = defaultdict(list)

    for i in range(n):
        dists = [(j, sq_Euclidean_d(dataset[i],dataset[j])) for j in range(n) if j != i]
        dists.sort(key=lambda x: x[1])
        knns_by_node[i] = dists[:k]

    return knns_by_node

def compute_empirical_geometry(dataset, traversable_radii_by_node, knns_by_node):
    """
    For each node v and each traversable radius idx (corresponding to a ball radius centered at v),
    determine the empirical radius needed to cover the kNNs of the boundary node u = u_subset[-1][0],
    and record the rank of the furthest of those neighbors from v.

    Returns:
        empirical_geometry: dict[int, dict[int, dict[str, float or int]]]
            Structure:
            empirical_geometry[v][idx] = {
                "radius": float,   # max distance from v to any of u's kNNs
                "rank": int        # rank of the furthest such kNN in v's distance order
            }
    """
    n = len(dataset)
    empirical_geometry = defaultdict(dict)

    for v in range(n):
        # Precompute all distances from v to others
        dists_v = [(j, sq_Euclidean_d(dataset[v],dataset[j])) for j in range(n) if j != v]
        dists_v.sort(key=lambda x: x[1])
        sorted_indices = [j for j, _ in dists_v]

        for idx, r in traversable_radii_by_node[v]:
            if idx >= len(dists_v):
                continue
            u = sorted_indices[idx]

            knns_u = [j for j, _ in knns_by_node[u]]
            dists_to_v = [(j, sq_Euclidean_d(dataset[v],dataset[j])) for j in knns_u]
            dists_to_v.sort(key=lambda x: x[1])

            furthest_idx, furthest_dist = dists_to_v[-1]
            try:
                rank = sorted_indices.index(furthest_idx)
            except ValueError:
                rank = -1  # Should not happen if dataset is consistent

            empirical_geometry[v][idx] = {
                "radius": furthest_dist, #distance between v and furthest KNN of u
                "index": furthest_idx, #index of furthest kNN of u
                "rank": rank #rank of u according to distance from v
            }

    return empirical_geometry

def compute_percentile_ranks(empirical_geometry, p):
    """
    For each traversable index (idx), compute the rank that is greater than or equal to 
    p-percent of the recorded ranks across all nodes.

    Args:
        empirical_geometry: dict[int, dict[int, dict[str, float or int]]]
            Structure: empirical_geometry[v][idx]["rank"]
        p: float in (0, 1), the desired percentile (e.g., 0.9 for 90%)

    Returns:
        percentile_rank_by_idx: dict[int, int]
            For each idx, the rank such that p% of the entries are ≤ that value.
    """

    rank_lists = defaultdict(list)

    # Collect all ranks by traversable index
    for v in empirical_geometry:
        for idx in empirical_geometry[v]:
            rank = empirical_geometry[v][idx]["rank"]
            if rank >= 0:
                rank_lists[idx].append(rank)

    # Compute the percentile rank per index
    percentile_rank_by_idx = {}
    for idx, ranks in rank_lists.items():
        if ranks:
            rank_threshold = int(np.percentile(ranks, 100 * p, interpolation='higher'))
            percentile_rank_by_idx[idx] = rank_threshold
        else:
            percentile_rank_by_idx[idx] = -1  # or np.inf / None if you prefer

    return percentile_rank_by_idx

def compute_percentile_radius_by_node(dataset, percentile_ranks):
    """
    For each node v and each traversable index, return the distance from v to its
    percentile-determined rank-th nearest neighbor.

    Args:
        dataset: np.ndarray of shape (n, d)
        percentile_ranks: dict[int, int] — for each traversable index, the desired rank

    Returns:
        percentile_radii_by_node: dict[int, dict[int, float]]
            Maps each node v to a dictionary mapping traversable idx → distance to rank-th NN
    """
    n = len(dataset)
    percentile_radii_by_node = defaultdict(dict)

    for v in range(n):
        # Precompute distances from v to all others, sorted
        dists = [
            (u, sq_Euclidean_d(dataset[v],dataset[u]))
            for u in range(n) if u != v
        ]
        dists.sort(key=lambda x: x[1])
        dist_list = [dist for _, dist in dists]

        for idx, rank in percentile_ranks.items():
            if rank < len(dist_list):
                percentile_radii_by_node[v][idx] = dist_list[rank]
            else:
                percentile_radii_by_node[v][idx] = float('inf')  # or np.nan

    return percentile_radii_by_node

def construction_phase(dataset, empirical=False, hnsw_M=32, ef_search=1):
    """
    Constructs FAISS HNSW graph and ball-traversable graph.

    Args:
        dataset: np.ndarray of shape (n, d)
        hnsw_M: int, max connections per node for HNSW
        ef_search: int, efSearch parameter during query

    Returns:
        index: FAISS HNSW index
        graph: dict[int, List[int]], ball-traversable graph
        traversable_radii_by_node: dict[int, List[float]], radius levels used during construction
    """
    d = dataset.shape[1]

    # Build FAISS HNSW index (optional dependency)
    try:
        import faiss  # moved import here to avoid hard dependency at module import time
    except Exception as _:
        faiss = None

    index = None
    if faiss is not None:
        index = faiss.IndexHNSWFlat(d, hnsw_M)
        index.hnsw.efSearch = ef_search
        index.hnsw.efConstruction = 100
        index.add(dataset)

    # Construct ball-traversable graph
    graph, traversable_radii_by_node = build_ball_traversable_graph(dataset)

    return index, graph, traversable_radii_by_node

def create_percentile_radii_by_node(dataset, traversable_radii_by_node, k=5, p=0.95):
    knns_by_node = compute_exact_knns(dataset, k=k)
    empirical_geometry = compute_empirical_geometry(dataset, traversable_radii_by_node, knns_by_node)
    percentile_ranks = compute_percentile_ranks(empirical_geometry, p=p)
    percentile_radii_by_node = compute_percentile_radius_by_node(dataset, percentile_ranks)
    return percentile_radii_by_node
