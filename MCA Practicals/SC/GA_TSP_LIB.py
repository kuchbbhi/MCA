"""
tsp_ga_lib.py

Reusable Genetic Algorithm library for the Traveling Salesman Problem (TSP).

Provides:
 - route_length, euclidean_distance
 - genetic operators: tournament_selection, ordered_crossover, swap_mutation
 - class GeneticTSP: configure & run GA, returns best route/distance/history
"""

import random
import math
from typing import List, Tuple, Optional

City = Tuple[float, float]
Route = List[int]


def euclidean_distance(a: City, b: City) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def route_length(route: Route, cities: List[City]) -> float:
    """Total closed-tour length of route (list of city indices)."""
    dist = 0.0
    n = len(route)
    for i in range(n):
        a = cities[route[i]]
        b = cities[route[(i + 1) % n]]
        dist += euclidean_distance(a, b)
    return dist


def create_route(n: int) -> Route:
    r = list(range(n))
    random.shuffle(r)
    return r


def initial_population(pop_size: int, n_cities: int) -> List[Route]:
    return [create_route(n_cities) for _ in range(pop_size)]


def tournament_selection(population: List[Route],
                         fitnesses: List[float],
                         k: int = 3) -> Route:
    """k-tournament selection; returns a copy of selected individual."""
    selected_idx = random.sample(range(len(population)), k)
    best = selected_idx[0]
    for idx in selected_idx[1:]:
        if fitnesses[idx] > fitnesses[best]:
            best = idx
    return population[best][:]


def ordered_crossover(parent1: Route, parent2: Route) -> Route:
    """Ordered crossover (OX) producing one child."""
    size = len(parent1)
    child = [-1] * size
    a, b = sorted(random.sample(range(size), 2))
    # copy slice from parent1
    child[a:b+1] = parent1[a:b+1]
    # fill remaining from parent2 in order
    p2_idx = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[p2_idx] in child:
                p2_idx += 1
            child[i] = parent2[p2_idx]
            p2_idx += 1
    return child


def swap_mutation(route: Route, mut_rate: float = 0.02) -> None:
    """In-place swap mutation: for each gene, with prob mut_rate swap with another."""
    size = len(route)
    for i in range(size):
        if random.random() < mut_rate:
            j = random.randrange(size)
            route[i], route[j] = route[j], route[i]


class GeneticTSP:
    """
    Genetic Algorithm for TSP.

    Example:
        ga = GeneticTSP(cities, pop_size=200, generations=500)
        best_route, best_dist, history = ga.run()
    """

    def __init__(self,
                 cities: List[City],
                 pop_size: int = 200,
                 generations: int = 500,
                 tournament_k: int = 5,
                 crossover_rate: float = 0.9,
                 mutation_rate: float = 0.02,
                 elitism: bool = True,
                 seed: Optional[int] = None):
        self.cities = cities
        self.n = len(cities)
        self.pop_size = pop_size
        self.generations = generations
        self.tournament_k = tournament_k
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        if seed is not None:
            random.seed(seed)

    def _compute_fitnesses(self, population: List[Route]):
        dists = [route_length(r, self.cities) for r in population]
        # fitness = inverse distance (avoid div-by-zero)
        fitnesses = [1.0 / (d + 1e-12) for d in dists]
        return fitnesses, dists

    def run(self, verbose: bool = False):
        """Run GA. Returns (best_route, best_distance, history_of_best_distance)."""
        # init
        population = initial_population(self.pop_size, self.n)
        fitnesses, dists = self._compute_fitnesses(population)

        best_idx = min(range(self.pop_size), key=lambda i: dists[i])
        best_route = population[best_idx][:]
        best_distance = dists[best_idx]
        history = [best_distance]

        for gen in range(1, self.generations + 1):
            new_pop: List[Route] = []

            # elitism: preserve best individual
            if self.elitism:
                new_pop.append(best_route[:])

            while len(new_pop) < self.pop_size:
                p1 = tournament_selection(population, fitnesses, k=self.tournament_k)
                p2 = tournament_selection(population, fitnesses, k=self.tournament_k)

                if random.random() < self.crossover_rate:
                    child = ordered_crossover(p1, p2)
                else:
                    child = p1[:]

                swap_mutation(child, mut_rate=self.mutation_rate)
                new_pop.append(child)

            population = new_pop[:self.pop_size]
            fitnesses, dists = self._compute_fitnesses(population)

            gen_best_idx = min(range(self.pop_size), key=lambda i: dists[i])
            gen_best_dist = dists[gen_best_idx]
            if gen_best_dist < best_distance:
                best_distance = gen_best_dist
                best_route = population[gen_best_idx][:]

            history.append(best_distance)
            if verbose and (gen <= 10 or gen % max(1, self.generations // 10) == 0):
                print(f"Gen {gen:4d} | Best distance: {best_distance:.6f}")

        return best_route, best_distance, history


# Optional helper to convert route -> coordinate sequence
def route_coords(route: Route, cities: List[City]) -> List[City]:
    return [cities[i] for i in route] + [cities[route[0]]]
