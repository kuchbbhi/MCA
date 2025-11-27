"""
run_tsp_ga.py

Example script showing how to use the tsp_ga_lib.GeneticTSP class.
Generates random cities, runs the GA, prints result, and plots progress & route.
This is a library which is to be provided to students in the lab examinations to execute the code
"""

import random
import matplotlib.pyplot as plt
from tsp_ga_lib import GeneticTSP, route_coords

def example_random_run(seed: int = 42):
    random.seed(seed)
    n_cities = 30
    cities = [(random.random() * 100, random.random() * 100) for _ in range(n_cities)]

    ga = GeneticTSP(
        cities=cities,
        pop_size=300,
        generations=500,
        tournament_k=5,
        crossover_rate=0.95,
        mutation_rate=0.02,
        elitism=True,
        seed=seed
    )

    best_route, best_distance, history = ga.run(verbose=True)

    print("\nBest distance:", best_distance)
    print("Best route (indices):", best_route)

    # Plot convergence
    plt.figure(figsize=(10,4))
    plt.plot(history)
    plt.title("GA progress: best distance per generation")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.grid(True)

    # Plot best route
    plt.figure(figsize=(6,6))
    coords = route_coords(best_route, cities)
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    plt.plot(xs, ys, marker='o')
    for i, city in enumerate(coords[:-1]):
        plt.text(city[0]+0.5, city[1]+0.5, str(best_route[i]), fontsize=8)
    plt.title(f"Best route (distance {best_distance:.3f})")
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    example_random_run()
