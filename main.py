"""
🏃 EvoSurvivor — Punto de entrada
Uso:
    python main.py                  # Ejecutar con visualización
    python main.py --headless       # Sin visualización (más rápido)
    python main.py --generations 50 # Número de generaciones
"""

import argparse
import sys
import os

# Asegurar que src está en el path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from genetic_algorithm import GeneticAlgorithm, NeuralNetwork
from game_engine import GameEnvironment


def train_headless(ga: GeneticAlgorithm, generations: int):
    """Entrena sin visualización — mucho más rápido."""
    print(f"\n{'='*50}")
    print("  🏃 EvoSurvivor — Modo Sin Visualización")
    print(f"{'='*50}\n")

    for gen in range(generations):
        population = ga.population
        nets = [NeuralNetwork(ind.weights) for ind in population]
        env = GameEnvironment(len(population))

        while True:
            alive = env.step(nets)
            if not alive:
                break

        scores = env.get_fitness_scores()
        for i, ind in enumerate(population):
            ind.fitness = scores[i]

        stats = ga.get_stats()
        print(
            f"Gen {stats['generation']:>4} │ "
            f"Mejor: {stats['best_current']:>7.0f} │ "
            f"Prom: {stats['avg_current']:>7.0f} │ "
            f"Récord: {stats['best_ever']:>7.0f}"
        )
        ga.evolve()

    ga.save("best_dino.json")
    print("\n✅ Entrenamiento completado. Mejor individuo guardado en best_dino.json")


def train_visual(ga: GeneticAlgorithm, speed: int):
    """Entrena con visualización Pygame."""
    from game_visual import GameVisualizer
    viz = GameVisualizer(ga, speed_multiplier=speed)
    viz.run()


def main():
    parser = argparse.ArgumentParser(description="EvoSurvivor con Algoritmo Genético")
    parser.add_argument("--headless", action="store_true", help="Sin visualización")
    parser.add_argument("--generations", type=int, default=100, help="Generaciones (headless)")
    parser.add_argument("--population", type=int, default=30, help="Tamaño de población")
    parser.add_argument("--mutation", type=float, default=0.1, help="Tasa de mutación")
    parser.add_argument("--speed", type=int, default=2, help="Multiplicador velocidad visual")
    args = parser.parse_args()

    ga = GeneticAlgorithm(
        population_size=args.population,
        mutation_rate=args.mutation,
        elitism=3,
        crossover_rate=0.8,
    )

    print(f"""
╔══════════════════════════════════════════╗
║    EvoSurvivor — Algoritmo Genético   ║
╠══════════════════════════════════════════╣
║  Población   : {args.population:<27}║
║  Mutación    : {args.mutation:<27}║
║  Elitismo    : 3                         ║
║  Red neuronal: 4→8→1 (tanh/sigmoid)     ║
╚══════════════════════════════════════════╝
""")

    if args.headless:
        train_headless(ga, args.generations)
    else:
        try:
            train_visual(ga, args.speed)
        except ImportError:
            print("⚠  Pygame no encontrado. Ejecutando en modo headless...")
            train_headless(ga, args.generations)


if __name__ == "__main__":
    main()