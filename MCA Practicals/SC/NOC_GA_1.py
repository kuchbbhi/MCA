import random, heapq, collections, itertools, json, os, tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---- helpers (mesh, OE, GA, surrogate) ----
def mesh_neighbors(x, y, W, H):
    for dx, dy, port in [(1, 0, "E"), (-1, 0, "W"), (0, 1, "N"), (0, -1, "S")]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < W and 0 <= ny < H:
            yield (nx, ny, port)


def mesh_links(W, H):
    links = []
    for y in range(H):
        for x in range(W):
            for nx, ny, _ in mesh_neighbors(x, y, W, H):
                links.append(((x, y), (nx, ny)))
    return links


def oe_legal(prev, curr, nxt):
    if prev is None:
        return True
    x, y = curr
    px, py = prev
    nx, ny = nxt
    dx1, dy1 = x - px, y - py
    dx2, dy2 = nx - x, ny - y
    if x % 2 == 0 and dx1 == 0 and dx2 == -1:
        return False
    if x % 2 == 1 and dx1 == 1 and dx2 != 0:
        return False
    return True


def random_chromosome(W, H, wmin=1, wmax=10):
    return {edge: random.randint(wmin, wmax) for edge in mesh_links(W, H)}


def shortest_path_oe(weights, W, H, s, d):
    INF = 10**9
    dist = {}
    prev = {}
    start = (s, None)
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        c, (u, pu) = heapq.heappop(pq)
        if u == d:
            path = [d]
            state = (d, pu)
            while state in prev:
                (u2, pu2) = prev[state]
                path.append(u2)
                state = (u2, pu2)
            return list(reversed(path))
        ux, uy = u
        for vx, vy, _ in mesh_neighbors(ux, uy, W, H):
            if pu is not None and not oe_legal(pu, u, (vx, vy)):
                continue
            w = weights.get(((ux, uy), (vx, vy)), 10)
            nv = ((vx, vy), u)
            nd = c + w
            if nv not in dist or nd < dist[nv]:
                dist[nv] = nd
                prev[nv] = (u, pu)
                heapq.heappush(pq, (nd, nv))
    return None


def decode_routes(weights, W, H, traffic_pairs):
    routes = {}
    for s, d in traffic_pairs:
        p = shortest_path_oe(weights, W, H, s, d)
        if p is None:
            return None
        routes[(s, d)] = p
    return routes


def simulate_surrogate(routes, inj_matrix, link_capacity=1.0):
    load = collections.defaultdict(float)
    for (s, d), path in routes.items():
        demand = inj_matrix.get((s, d), 0.0)
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            load[(u, v)] += demand
    max_util = 0.0
    delay_sum = 0.0
    demand_sum = 0.0
    for (u, v), l in load.items():
        util = l / link_capacity
        if util >= 1.0:
            return 1e9 + 1e6 * (util - 1.0), load
        max_util = max(max_util, util)
        if l > 0:
            delay_sum += l * (1.0 / (1.0 - util))
            demand_sum += l
    avg_delay = (delay_sum / demand_sum) if demand_sum > 0 else 0.0
    energy = sum(load.values())
    fitness = 0.6 * avg_delay + 0.3 * max_util + 0.1 * energy
    return fitness, load


def crossover(a, b, px=0.5):
    c = {}
    for k in a:
        c[k] = a[k] if random.random() < px else b[k]
    return c


def mutate(ch, p=0.08, step=1, wmin=1, wmax=10):
    new = ch.copy()
    for k in list(new.keys()):
        if random.random() < p:
            new[k] = max(wmin, min(wmax, new[k] + random.choice([-step, -1, 1, step])))
    return new


def make_population(n, W, H):
    return [random_chromosome(W, H) for _ in range(n)]


def evolve_and_track(W=4, H=4, pop_size=20, gens=20, elite=3):
    pairs = []
    inj = {}
    base_demand = 0.005
    for y1 in range(H):
        for x1 in range(W):
            for y2 in range(H):
                for x2 in range(W):
                    if (x1, y1) != (x2, y2):
                        pairs.append(((x1, y1), (x2, y2)))
                        inj[((x1, y1), (x2, y2))] = base_demand
    pop = make_population(pop_size, W, H)
    history = {"best_fitness": [], "avg_fitness": [], "best_chrom": None}
    best_overall = (1e18, None, None)
    for g in range(gens):
        scored = []
        for ch in pop:
            routes = decode_routes(ch, W, H, pairs)
            if routes is None:
                fitness = 1e12
                load = {}
            else:
                fitness, load = simulate_surrogate(routes, inj, link_capacity=1.0)
            scored.append((fitness, ch, routes, load))
        scored.sort(key=lambda x: x[0])
        fit_vals = [s[0] for s in scored]
        history["best_fitness"].append(fit_vals[0])
        history["avg_fitness"].append(sum(fit_vals) / len(fit_vals))
        if scored[0][0] < best_overall[0]:
            best_overall = (scored[0][0], scored[0][3], scored[0][2])
            history["best_chrom"] = scored[0][1]
        newpop = [scored[i][1] for i in range(min(elite, len(scored)))]

        def tournament():
            a, b = random.sample(scored, 2)
            return a if a[0] < b[0] else b

        while len(newpop) < pop_size:
            p1 = tournament()[1]
            p2 = tournament()[1]
            child = crossover(p1, p2, px=0.5)
            child = mutate(child, p=0.08, step=1)
            newpop.append(child)
        pop = newpop
    return history, best_overall


# ---- run & plot ----
random.seed(42)
history, best = evolve_and_track(W=4, H=4, pop_size=20, gens=20, elite=3)

plt.figure()
plt.plot(history["best_fitness"], label="best fitness")
plt.plot(history["avg_fitness"], label="avg fitness")
plt.xlabel("Generation")
plt.ylabel("Fitness (lower better)")
plt.title("GA Fitness over Generations (4x4 demo)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

best_fitness, best_loads, best_routes = best[0], best[1], best[2]
if best_loads is None:
    print("No valid solution found.")
else:
    rows = [
        {"from": str(u), "to": str(v), "util": float(l)}
        for (u, v), l in best_loads.items()
    ]
    df_links = pd.DataFrame(rows)
    print(df_links.head(12).to_string())

    plt.figure()
    plt.hist(df_links["util"], bins=15)
    plt.xlabel("Utilization (flits/cycle)")
    plt.ylabel("Number of directed links")
    plt.title("Histogram of directed link utilizations (best solution)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # heatmaps (undirected combined)
    W, H = 4, 4
    undirected = collections.defaultdict(float)
    for (u, v), l in best_loads.items():
        a = tuple(u)
        b = tuple(v)
        key = tuple(sorted([a, b]))
        undirected[key] += l
    import numpy as np

    horiz = np.zeros((H, W - 1))
    vert = np.zeros((H - 1, W))
    for (a, b), val in undirected.items():
        (ax, ay), (bx, by) = a, b
        if ay == by:
            x = min(ax, bx)
            y = ay
            horiz[y, x] = val
        elif ax == bx:
            x = ax
            y = min(ay, by)
            vert[y, x] = val
    plt.figure()
    plt.imshow(horiz, aspect="auto")
    plt.title("Horizontal edge loads")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    plt.figure()
    plt.imshow(vert, aspect="auto")
    plt.title("Vertical edge loads")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


# --- Safe JSON export ---
def safe_serialize(obj):
    """Recursively convert tuples to strings for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize(x) for x in obj]
    else:
        return obj


summary = {
    "history": safe_serialize(history),
    "best_fitness_value": float(best_fitness),
    "best_loads": safe_serialize(best_loads),
    "sample_routes": safe_serialize(dict(list(best_routes.items())[:10])),
}

tmpfile = os.path.join(tempfile.gettempdir(), "noc_ga_summary_fixed.json")
with open(tmpfile, "w") as f:
    json.dump(summary, f, indent=2)

print(f"✅ Saved serializable summary to: {tmpfile}")
