# noc_ga_compact.py
"""
Compact runner that uses noc_ga_lib.py.
Produces the same plots + JSON summary as before, but the logic is delegated
to the library for clarity and reuse.
"""

import random
import json
import os
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#THIS LIB IS THE PRIMARY KEY FOR THE OPTIMIZED CODE
from noc_ga_lib import evolve_and_track_compact, save_summary

# ----------------------
# user parameters
# ----------------------
W, H = 4, 4
POP_SIZE = 20
GENS = 20
ELITE = 3
SEED = 42

# ----------------------
# run GA (library does the heavy lifting)
# ----------------------
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    history, best, mesh_data = evolve_and_track_compact(
        W=W, H=H, pop_size=POP_SIZE, gens=GENS, elite=ELITE, seed=SEED
    )

    best_fitness, best_loads, best_routes = best
    _, _, edges = mesh_data

    # --- fitness plot ---
    plt.figure()
    plt.plot(history['best_fitness'], label='best fitness')
    plt.plot(history['avg_fitness'], label='avg fitness')
    plt.xlabel('Generation'); plt.ylabel('Fitness (lower better)')
    plt.title(f'GA Fitness over Generations ({W}x{H} demo)')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # --- show best loads as DataFrame ---
    if best_loads is None:
        print("No valid solution found.")
    else:
        rows = [{'from': str(u), 'to': str(v), 'util': float(val)}
                for (u, v), val in zip(edges, best_loads)]
        df_links = pd.DataFrame(rows)
        print(df_links.head(12).to_string())

        # histogram
        plt.figure()
        plt.hist(df_links['util'], bins=15)
        plt.xlabel('Utilization (flits/cycle)')
        plt.ylabel('Number of directed links')
        plt.title('Histogram of directed link utilizations (best solution)')
        plt.grid(True); plt.tight_layout(); plt.show()

        # heatmaps (undirected aggregated)
        undirected = {}
        for (u, v), val in zip(edges, best_loads):
            key = tuple(sorted([u, v]))
            undirected[key] = undirected.get(key, 0.0) + float(val)

        coords = [(i % W, i // W) for i in range(W * H)]
        import numpy as _np
        horiz = _np.zeros((H, W - 1))
        vert = _np.zeros((H - 1, W))
        for (a, b), val in undirected.items():
            ax, ay = coords[a]
            bx, by = coords[b]
            if ay == by:
                x = min(ax, bx); y = ay; horiz[y, x] = val
            elif ax == bx:
                x = ax; y = min(ay, by); vert[y, x] = val

        plt.figure(); plt.imshow(horiz, aspect='auto'); plt.title('Horizontal edge loads'); plt.colorbar(); plt.tight_layout(); plt.show()
        plt.figure(); plt.imshow(vert, aspect='auto'); plt.title('Vertical edge loads'); plt.colorbar(); plt.tight_layout(); plt.show()

    # --- save summary via library helper ---
    summary_path = save_summary(history, best)
    print(f"✅ Saved serializable summary to: {summary_path}")

if __name__ == "__main__":
    main()
