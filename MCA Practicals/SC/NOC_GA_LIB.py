"""noc_ga_lib.py

Compact NoC GA utilities module.

Provides functions to build a mesh, compute OE-legal shortest paths, decode routes,
simulate a surrogate network (loads & fitness), and run a simple GA to optimize link
weights. Designed for teaching: small, well-documented functions and a single
high-level `evolve_and_track_compact` entrypoint.

Example usage (in another script or REPL):

    from noc_ga_lib import evolve_and_track_compact
    history, best, mesh = evolve_and_track_compact(W=4, H=4, pop_size=20, gens=20)

The module is intentionally self-contained and has no side-effects on import.
"""

from typing import List, Tuple, Dict, Optional
import random
import heapq
import json
import os
import tempfile

import numpy as np

Coord = Tuple[int, int]
Edge = Tuple[int, int]

# ----------------------
# mesh helpers (compact)
# ----------------------

def coords_from_index(idx: int, W: int) -> Coord:
    return (idx % W, idx // W)


def index_from_coords(x: int, y: int, W: int) -> int:
    return y * W + x


def build_mesh(W: int, H: int):
    """Build directed mesh graph.

    Returns:
      - N: number of nodes
      - coords: list of (x,y) for each node index
      - edges: list of directed edges (u,v) where u and v are node indices
      - adj: adjacency list mapping node -> list of (neighbor, edge_index)
    """
    N = W * H
    coords = [coords_from_index(i, W) for i in range(N)]
    edges: List[Edge] = []
    adj: List[List[Tuple[int,int]]] = [[] for _ in range(N)]
    for y in range(H):
        for x in range(W):
            u = index_from_coords(x, y, W)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:  # E W N S
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    v = index_from_coords(nx, ny, W)
                    edge_idx = len(edges)
                    edges.append((u, v))
                    adj[u].append((v, edge_idx))
    return N, coords, edges, adj


def oe_legal_by_coords(prev: Optional[int], curr: int, nxt: int, coords: List[Coord]) -> bool:
    """OE legality check using coordinates. Matches the logic in the demo code.

    prev == None is represented by prev == -1 in code that calls this.
    """
    if prev is None or prev == -1:
        return True
    x, y = coords[curr]
    px, py = coords[prev]
    nx, ny = coords[nxt]
    dx1, dy1 = x - px, y - py
    dx2, dy2 = nx - x, ny - y
    if x % 2 == 0 and dx1 == 0 and dx2 == -1:
        return False
    if x % 2 == 1 and dx1 == 1 and dx2 != 0:
        return False
    return True

# ----------------------
# shortest path with OE
# ----------------------

def shortest_path_oe(chrom_weights: np.ndarray, N: int, adj: List[List[Tuple[int,int]]],
                     coords: List[Coord], src: int, dst: int) -> Optional[List[int]]:
    """Dijkstra over state (node, prev_node). Return node path or None if unreachable.

    prev is encoded as -1 for "None".
    """
    INF = 10**12
    dist = {}
    prev_state = {}
    start_state = (src, -1)
    dist[start_state] = 0.0
    pq = [(0.0, start_state)]
    while pq:
        dcur, (u, pu) = heapq.heappop(pq)
        if dcur > dist.get((u, pu), INF):
            continue
        if u == dst:
            # reconstruct
            path = [u]
            state = (u, pu)
            while state in prev_state:
                u2, pu2 = prev_state[state]
                path.append(u2)
                state = (u2, pu2)
            return list(reversed(path))
        for v, eidx in adj[u]:
            if pu != -1 and not oe_legal_by_coords(pu, u, v, coords):
                continue
            w = float(chrom_weights[eidx])
            ns = (v, u)
            nd = dcur + w
            if nd < dist.get(ns, INF):
                dist[ns] = nd
                prev_state[ns] = (u, pu)
                heapq.heappush(pq, (nd, ns))
    return None

# ----------------------
# routing & surrogate
# ----------------------

def decode_routes_for_pairs(chrom: np.ndarray, W: int, H: int,
                            pairs: List[Tuple[int,int]],
                            N: int, adj, coords) -> Optional[Dict[Tuple[int,int], List[int]]]:
    """Decode shortest OE-legal routes for a list of (s,d) pairs using chrom weights.

    Returns None if any pair is unreachable.
    """
    routes = {}
    for s, d in pairs:
        p = shortest_path_oe(chrom, N, adj, coords, s, d)
        if p is None:
            return None
        routes[(s, d)] = p
    return routes


def simulate_surrogate(routes: Dict[Tuple[int,int], List[int]],
                       inj: Dict[Tuple[int,int], float],
                       edges: List[Edge],
                       link_capacity: float = 1.0):
    """Compute directed edge loads, evaluate fitness.

    Fitness = 0.6*avg_delay + 0.3*max_util + 0.1*energy
    If any link util >= 1.0, returns a large penalty proportional to overflow.
    Returns (fitness, loads_array).
    """
    edge_index_map = {e: idx for idx, e in enumerate(edges)}
    loads = np.zeros(len(edges), dtype=float)
    for (s, d), path in routes.items():
        demand = inj.get((s, d), 0.0)
        for i in range(len(path) - 1):
            e = (path[i], path[i + 1])
            ei = edge_index_map[e]
            loads[ei] += demand
    utils = loads / link_capacity
    if np.any(utils >= 1.0):
        overflow = float(np.max(utils) - 1.0)
        return 1e9 + 1e6 * overflow, loads
    max_util = float(np.max(utils)) if loads.size else 0.0
    demand_sum = float(np.sum(loads))
    if demand_sum > 0:
        delay_sum = float(np.sum(loads * (1.0 / (1.0 - utils + 1e-12))))
        avg_delay = delay_sum / demand_sum
    else:
        avg_delay = 0.0
    energy = float(np.sum(loads))
    fitness = 0.6 * avg_delay + 0.3 * max_util + 0.1 * energy
    return fitness, loads

# ----------------------
# GA helpers (numpy-friendly)
# ----------------------

def random_population(pop_size: int, n_edges: int, wmin=1, wmax=10):
    return np.random.randint(wmin, wmax + 1, size=(pop_size, n_edges), dtype=np.int32)


def tournament_select(scored: List[Tuple[float, int]], k=2):
    a, b = random.sample(scored, k)
    return a if a[0] < b[0] else b


def crossover_np(p1: np.ndarray, p2: np.ndarray, px=0.5):
    mask = np.random.rand(p1.size) < px
    child = p1.copy()
    child[~mask] = p2[~mask]
    return child


def mutate_np(chrom: np.ndarray, p=0.08, step=1, wmin=1, wmax=10):
    mask = np.random.rand(chrom.size) < p
    if not mask.any():
        return chrom
    changes = np.random.choice([-step, -1, 1, step], size=mask.sum())
    new = chrom.copy()
    new[mask] = np.clip(new[mask] + changes, wmin, wmax)
    return new

# ----------------------
# main GA loop
# ----------------------

def evolve_and_track_compact(W=4, H=4, pop_size=20, gens=20, elite=3, seed: Optional[int] = None):
    """Run GA to optimize link weights.

    Returns:
      - history: dict with lists 'best_fitness' and 'avg_fitness' and 'best_chrom'
      - best_overall: tuple (fitness, loads_array, routes)
      - mesh_info: (W,H,edges)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    N, coords, edges, adj = build_mesh(W, H)
    n_edges = len(edges)
    pairs = [(s, d) for s in range(N) for d in range(N) if s != d]
    base_demand = 0.005
    inj = {pair: base_demand for pair in pairs}

    pop = random_population(pop_size, n_edges, wmin=1, wmax=10)

    history = {"best_fitness": [], "avg_fitness": [], "best_chrom": None}
    best_overall = (1e18, None, None)

    for g in range(gens):
        scored = []
        for i in range(pop.shape[0]):
            chrom = pop[i]
            routes = decode_routes_for_pairs(chrom, W, H, pairs, N, adj, coords)
            if routes is None:
                fitness = 1e12
                loads = None
            else:
                fitness, loads = simulate_surrogate(routes, inj, edges)
            scored.append((fitness, i, chrom.copy(), routes, loads))
        scored.sort(key=lambda x: x[0])
        fits = [s[0] for s in scored]
        history["best_fitness"].append(fits[0])
        history["avg_fitness"].append(float(np.mean(fits)))
        if scored[0][0] < best_overall[0]:
            best_overall = (scored[0][0], scored[0][4], scored[0][3])
            history["best_chrom"] = scored[0][2].copy()

        newpop = [scored[i][2].copy() for i in range(min(elite, len(scored)))]
        pool = [(s[0], s[1]) for s in scored]
        while len(newpop) < pop_size:
            p1 = tournament_select(pool)[1]
            p2 = tournament_select(pool)[1]
            child = crossover_np(pop[p1], pop[p2], px=0.5)
            child = mutate_np(child, p=0.08, step=1)
            newpop.append(child)
        pop = np.stack(newpop, axis=0)

    return history, best_overall, (W, H, edges)

# ----------------------
# small convenience: save summary
# ----------------------

def save_summary(history, best, filename=None):
    """Save a JSON summary of history and best solution. Returns the path."""
    def safe_serialize(obj):
        if isinstance(obj, dict):
            return {str(k): safe_serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, np.ndarray)):
            return [safe_serialize(x) for x in obj]
        else:
            return obj

    best_fitness = float(best[0])
    best_loads = best[1].tolist() if best[1] is not None else None
    best_routes = dict(list(best[2].items())[:10]) if best[2] else {}

    summary = {
        'history': safe_serialize(history),
        'best_fitness_value': best_fitness,
        'best_loads': safe_serialize(best_loads),
        'sample_routes': safe_serialize(best_routes)
    }
    if filename is None:
        filename = os.path.join(tempfile.gettempdir(), 'noc_ga_lib_summary.json')
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)
    return filename


__all__ = [
    'build_mesh', 'oe_legal_by_coords', 'shortest_path_oe',
    'decode_routes_for_pairs', 'simulate_surrogate',
    'random_population', 'crossover_np', 'mutate_np',
    'evolve_and_track_compact', 'save_summary'
]
