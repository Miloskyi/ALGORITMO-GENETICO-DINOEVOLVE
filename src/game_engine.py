"""
Motor del juego - Física del dinosaurio y obstáculos
Sin dependencia de pygame aquí; el renderizado lo hace game_visual.py
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ── Constantes de física ───────────────────────────────────────────────────────
SCREEN_W = 900
SCREEN_H = 500
GROUND_Y = 320
GRAVITY = 1.2
JUMP_VELOCITY = -18
INITIAL_SPEED = 6.0
SPEED_INCREMENT = 0.0015      # Velocidad aumenta con el tiempo


@dataclass
class Dino:
    x: float = 80
    y: float = GROUND_Y
    vy: float = 0.0           # Velocidad vertical
    on_ground: bool = True
    alive: bool = True
    width: int = 44
    height: int = 48
    score: float = 0.0

    def jump(self):
        if self.on_ground:
            self.vy = JUMP_VELOCITY
            self.on_ground = False

    def update(self):
        if not self.alive:
            return
        self.vy += GRAVITY
        self.y += self.vy
        if self.y >= GROUND_Y:
            self.y = GROUND_Y
            self.vy = 0
            self.on_ground = True
        self.score += 1

    @property
    def rect(self):
        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass
class Cactus:
    x: float
    width: int = 20
    height: int = 0
    speed: float = INITIAL_SPEED

    def __post_init__(self):
        # Cactus de diferentes alturas
        self.height = random.randint(30, 65)
        self.y = GROUND_Y + 48 - self.height   # Anclado al piso

    def update(self):
        self.x -= self.speed

    @property
    def rect(self):
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def off_screen(self) -> bool:
        return self.x + self.width < 0


def rects_collide(r1, r2, margin: int = 6) -> bool:
    """Colisión AABB con margen de tolerancia."""
    x1, y1, x1b, y1b = r1
    x2, y2, x2b, y2b = r2
    m = margin
    return not (x1b - m < x2 or x1 + m > x2b or y1b - m < y2 or y1 + m > y2b)


class GameEnvironment:
    """
    Entorno de simulación para N dinosaurios (una generación completa).
    """
    def __init__(self, n_dinos: int):
        self.n_dinos = n_dinos
        self.dinos: List[Dino] = [Dino() for _ in range(n_dinos)]
        self.cacti: List[Cactus] = []
        self.speed = INITIAL_SPEED
        self.frame = 0
        self.next_cactus_frame = self._random_gap()

    def _random_gap(self) -> int:
        """Frames hasta el siguiente cactus."""
        return random.randint(55, 130)

    def _get_inputs(self, dino: Dino) -> np.ndarray:
        """
        Extrae las 4 características que ve la red neuronal.
        Todo normalizado entre 0 y 1.
        """
        if self.cacti:
            # Cactus más cercano adelante del dino
            ahead = [c for c in self.cacti if c.x + c.width > dino.x]
            if ahead:
                c = min(ahead, key=lambda c: c.x)
                dist = max(0, c.x - dino.x) / SCREEN_W
                height_norm = c.height / 70
            else:
                dist, height_norm = 1.0, 0.5
        else:
            dist, height_norm = 1.0, 0.5

        speed_norm = (self.speed - INITIAL_SPEED) / 8
        dino_height = (GROUND_Y - dino.y) / (GROUND_Y * 0.6)
        return np.array([dist, height_norm, speed_norm, dino_height], dtype=float)

    def step(self, neural_nets) -> bool:
        """
        Avanza un frame. Retorna True si quedan dinos vivos.
        neural_nets: lista de NeuralNetwork, uno por dino.
        """
        self.frame += 1
        self.speed = INITIAL_SPEED + self.frame * SPEED_INCREMENT

        # Generar cactus
        if self.frame >= self.next_cactus_frame:
            self.cacti.append(Cactus(x=SCREEN_W + 10, speed=self.speed))
            self.next_cactus_frame = self.frame + self._random_gap()

        # Actualizar cactus
        for c in self.cacti:
            c.speed = self.speed
            c.update()
        self.cacti = [c for c in self.cacti if not c.off_screen()]

        # Actualizar dinos y aplicar IA
        any_alive = False
        for i, dino in enumerate(self.dinos):
            if not dino.alive:
                continue
            any_alive = True

            inputs = self._get_inputs(dino)
            decision = neural_nets[i].predict(inputs)
            if decision > 0.5:
                dino.jump()

            dino.update()

            # Colisión
            for c in self.cacti:
                if rects_collide(dino.rect, c.rect):
                    dino.alive = False
                    break

        return any_alive

    def get_fitness_scores(self) -> List[float]:
        return [d.score for d in self.dinos]

    @property
    def alive_count(self) -> int:
        return sum(1 for d in self.dinos if d.alive)