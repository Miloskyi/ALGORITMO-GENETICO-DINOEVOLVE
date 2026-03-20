"""
Algoritmo Genético - Núcleo
Optimiza los pesos de una red neuronal simple para que el dinosaurio aprenda a saltar.
"""

import numpy as np
import random
import json
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Individual:
    """Representa un dinosaurio con su 'cerebro' (pesos de red neuronal)."""
    weights: np.ndarray          # Pesos de la red neuronal
    fitness: float = 0.0        # Qué tan lejos llegó
    alive: bool = True
    distance: float = 0.0

    def copy(self) -> "Individual":
        return Individual(weights=self.weights.copy(), fitness=self.fitness)


class NeuralNetwork:
    """
    Red neuronal simple: 4 entradas → 8 oculta → 1 salida
    Entradas: [distancia_cactus, altura_cactus, velocidad, altura_dino]
    Salida: probabilidad de saltar (> 0.5 = salta)
    """
    INPUT_SIZE = 4
    HIDDEN_SIZE = 8
    OUTPUT_SIZE = 1
    TOTAL_WEIGHTS = (INPUT_SIZE * HIDDEN_SIZE) + HIDDEN_SIZE + (HIDDEN_SIZE * OUTPUT_SIZE) + OUTPUT_SIZE

    def __init__(self, weights: np.ndarray = None):
        if weights is None:
            weights = np.random.uniform(-1, 1, self.TOTAL_WEIGHTS)
        self.weights = weights
        self._unpack_weights()

    def _unpack_weights(self):
        idx = 0
        s = self.INPUT_SIZE * self.HIDDEN_SIZE
        self.W1 = self.weights[idx:idx+s].reshape(self.INPUT_SIZE, self.HIDDEN_SIZE)
        idx += s
        self.b1 = self.weights[idx:idx+self.HIDDEN_SIZE]
        idx += self.HIDDEN_SIZE
        s2 = self.HIDDEN_SIZE * self.OUTPUT_SIZE
        self.W2 = self.weights[idx:idx+s2].reshape(self.HIDDEN_SIZE, self.OUTPUT_SIZE)
        idx += s2
        self.b2 = self.weights[idx:idx+self.OUTPUT_SIZE]

    def predict(self, inputs: np.ndarray) -> float:
        hidden = np.tanh(inputs @ self.W1 + self.b1)
        output = 1 / (1 + np.exp(-(hidden @ self.W2 + self.b2)))
        return float(output[0])


class GeneticAlgorithm:
    """
    Algoritmo Genético que evoluciona la población de dinosaurios.
    """
    def __init__(
        self,
        population_size: int = 30,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.3,
        elitism: int = 3,
        crossover_rate: float = 0.8,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elitism = elitism
        self.crossover_rate = crossover_rate

        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.population: List[Individual] = []

        self._init_population()

    def _init_population(self):
        self.population = [
            Individual(weights=np.random.uniform(-1, 1, NeuralNetwork.TOTAL_WEIGHTS))
            for _ in range(self.population_size)
        ]

    # ── Selección por torneo ──────────────────────────────────────────────────
    def _tournament_selection(self, k: int = 3) -> Individual:
        contenders = random.sample(self.population, k)
        return max(contenders, key=lambda ind: ind.fitness)

    # ── Cruce (crossover de un punto) ─────────────────────────────────────────
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        point = random.randint(1, len(parent1.weights) - 1)
        w1 = np.concatenate([parent1.weights[:point], parent2.weights[point:]])
        w2 = np.concatenate([parent2.weights[:point], parent1.weights[point:]])
        return Individual(weights=w1), Individual(weights=w2)

    # ── Mutación gaussiana ────────────────────────────────────────────────────
    def _mutate(self, individual: Individual) -> Individual:
        weights = individual.weights.copy()
        mask = np.random.random(len(weights)) < self.mutation_rate
        weights[mask] += np.random.normal(0, self.mutation_strength, mask.sum())
        weights = np.clip(weights, -2, 2)
        return Individual(weights=weights)

    # ── Evolucionar a la siguiente generación ─────────────────────────────────
    def evolve(self) -> List[Individual]:
        # Ordenar por fitness descendente
        sorted_pop = sorted(self.population, key=lambda i: i.fitness, reverse=True)

        fitnesses = [ind.fitness for ind in sorted_pop]
        best = fitnesses[0]
        avg = np.mean(fitnesses)
        self.best_fitness_history.append(best)
        self.avg_fitness_history.append(avg)

        new_population: List[Individual] = []

        # Elitismo: los mejores pasan directo
        for i in range(self.elitism):
            new_population.append(sorted_pop[i].copy())

        # Reproducción
        while len(new_population) < self.population_size:
            p1 = self._tournament_selection()
            p2 = self._tournament_selection()
            child1, child2 = self._crossover(p1, p2)
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            new_population.extend([child1, child2])

        self.population = new_population[:self.population_size]
        self.generation += 1

        # Reset para nueva generación
        for ind in self.population:
            ind.fitness = 0.0
            ind.alive = True
            ind.distance = 0.0

        return self.population

    def get_stats(self) -> dict:
        fitnesses = [ind.fitness for ind in self.population]
        alive = sum(1 for ind in self.population if ind.alive)
        return {
            "generation": self.generation,
            "alive": alive,
            "best_current": max(fitnesses),
            "avg_current": np.mean(fitnesses),
            "best_ever": max(self.best_fitness_history) if self.best_fitness_history else 0,
            "history_best": self.best_fitness_history,
            "history_avg": self.avg_fitness_history,
        }

    def save(self, path: str):
        best = max(self.population, key=lambda i: i.fitness)
        data = {
            "generation": self.generation,
            "best_weights": best.weights.tolist(),
            "best_fitness": best.fitness,
            "history_best": self.best_fitness_history,
            "history_avg": self.avg_fitness_history,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Guardado en {path}")