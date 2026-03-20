# 🦕 Dino GA — Dinosaurio con Algoritmo Genético

> Proyecto de optimización con Algoritmo Genético aplicado al juego del dinosaurio de Google Chrome.

---

## 📁 Estructura del Proyecto

```
dino_ga/
├── main.py                  # Punto de entrada principal
├── requirements.txt         # Dependencias
├── best_dino.json           # Se genera al guardar (tecla S)
│
└── src/
    ├── genetic_algorithm.py # Núcleo del AG (población, selección, cruce, mutación)
    ├── game_engine.py       # Física del juego (dinos, cactus, colisiones)
    └── game_visual.py       # Visualización con Pygame
```

---

## 🧬 Cómo Funciona el Algoritmo Genético

### Problema de Optimización
**Objetivo**: encontrar los pesos óptimos de una red neuronal que maximice la distancia recorrida por el dinosaurio sin chocar con cactus.

### Representación (Cromosoma)
Cada individuo = vector de **pesos reales** de una red neuronal 4→8→1:
- `4 entradas × 8 neuronas + 8 bias + 8 × 1 salida + 1 bias = 81 genes`

### Entradas de la Red Neuronal
| # | Entrada | Rango |
|---|---------|-------|
| 1 | Distancia al cactus más cercano | [0, 1] |
| 2 | Altura del cactus | [0, 1] |
| 3 | Velocidad del juego | [0, 1] |
| 4 | Altura actual del dinosaurio | [0, 1] |

### Función de Fitness
```
fitness(individuo) = frames_sobrevividos
```
Cuanto más lejos llegue el dino, mayor su fitness.

### Operadores Genéticos

#### Selección — Torneo (k=3)
Se eligen 3 individuos al azar y el de mayor fitness pasa a reproducirse.

#### Cruce — Un Punto
```
Padre 1: [w1, w2, w3, | w4, w5, w6]
Padre 2: [a1, a2, a3, | a4, a5, a6]
Hijo 1:  [w1, w2, w3,   a4, a5, a6]  ✓
```

#### Mutación — Gaussiana
Con probabilidad `mutation_rate`, se añade ruido gaussiano `N(0, σ)` a cada gen.

#### Elitismo
Los 3 mejores individuos de cada generación pasan intactos a la siguiente.

---

## 🚀 Instalación y Uso

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar con visualización (recomendado)
python main.py

# 3. Ejecutar sin visualización (más rápido)
python main.py --headless --generations 200

# 4. Opciones avanzadas
python main.py --population 50 --mutation 0.15 --speed 3
```

### Controles en Pantalla
| Tecla | Acción |
|-------|--------|
| `↑` | Aumentar velocidad de simulación |
| `↓` | Reducir velocidad de simulación |
| `S` | Guardar el mejor individuo |

---

## 📊 Parámetros del AG

| Parámetro | Valor por defecto | Descripción |
|-----------|------------------|-------------|
| `population_size` | 30 | Individuos por generación |
| `mutation_rate` | 0.10 | Probabilidad de mutar cada gen |
| `mutation_strength` | 0.30 | Desviación estándar del ruido |
| `crossover_rate` | 0.80 | Probabilidad de cruce |
| `elitism` | 3 | Mejores que sobreviven intactos |

---

## 🏗️ Arquitectura del Sistema

```
main.py
  │
  ├─► GeneticAlgorithm          ← Gestiona la evolución
  │     ├── population[]         ← Lista de Individual
  │     ├── _tournament_selection()
  │     ├── _crossover()
  │     ├── _mutate()
  │     └── evolve()             ← Produce nueva generación
  │
  ├─► NeuralNetwork             ← "Cerebro" de cada dino
  │     └── predict(inputs) → float
  │
  ├─► GameEnvironment           ← Simula la física
  │     ├── step(nets) → bool
  │     └── get_fitness_scores()
  │
  └─► GameVisualizer (opcional) ← Renderiza con Pygame
```