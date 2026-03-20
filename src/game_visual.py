"""
Visualizador con Pygame
Renderiza el juego y las estadísticas del AG en tiempo real.
"""

import pygame
import sys
import math
import json
import threading
from pathlib import Path
from typing import List, Optional

from game_engine import GameEnvironment, SCREEN_W, SCREEN_H, GROUND_Y, Dino, Cactus
from genetic_algorithm import GeneticAlgorithm, NeuralNetwork, Individual

# ── Paleta de colores ─────────────────────────────────────────────────────────
BG          = (15,  17,  26)
BG2         = (22,  26,  40)
GROUND_CLR  = (50,  55,  80)
DINO_DEAD   = (60,  65,  90)
DINO_ALIVE  = (100, 220, 160)
DINO_BEST   = (255, 220,  60)
CACTUS_CLR  = (80,  190, 130)
ACCENT      = (100, 130, 255)
TEXT_CLR    = (200, 210, 240)
DIM_TEXT    = (90,  100, 130)
GRAPH_LINE  = (100, 130, 255)
GRAPH_AVG   = (80,  200, 150)
RED_CLR     = (255,  80,  80)

FPS = 60
PANEL_W = 260    # Ancho del panel lateral de stats


def draw_dino(surf: pygame.Surface, dino: Dino, color):
    """Dibuja el dinosaurio como formas geométricas estilizadas."""
    x, y = int(dino.x), int(dino.y)
    w, h = dino.width, dino.height

    alpha_surf = pygame.Surface((w + 10, h + 10), pygame.SRCALPHA)

    # Cuerpo
    pygame.draw.rect(alpha_surf, (*color, 200), (5, h//3, w - 8, h * 2//3), border_radius=4)
    # Cabeza
    pygame.draw.rect(alpha_surf, (*color, 220), (w//3, 0, w * 2//3, h//2), border_radius=6)
    # Ojo
    eye_color = (15, 17, 26) if dino.alive else (40, 44, 60)
    pygame.draw.circle(alpha_surf, eye_color, (w - 4, h//5), 5)
    # Patas (animadas)
    leg_offset = int(math.sin(pygame.time.get_ticks() * 0.015) * 5) if dino.on_ground and dino.alive else 0
    pygame.draw.rect(alpha_surf, (*color, 180), (8, h - 10 + leg_offset, 8, 12))
    pygame.draw.rect(alpha_surf, (*color, 180), (22, h - 10 - leg_offset, 8, 12))

    surf.blit(alpha_surf, (x - 5, y - 5))


def draw_cactus(surf: pygame.Surface, cactus: Cactus):
    x, y = int(cactus.x), int(cactus.y)
    w, h = cactus.width, cactus.height
    color = CACTUS_CLR

    # Tronco principal
    pygame.draw.rect(surf, color, (x + w//3, y, w//3, h), border_radius=3)
    # Brazos
    arm_y = y + h//3
    pygame.draw.rect(surf, color, (x, arm_y - 8, w//3 + 2, 7), border_radius=2)
    pygame.draw.rect(surf, color, (x, arm_y - 18, 7, 12), border_radius=2)
    arm2_y = y + h//2
    pygame.draw.rect(surf, color, (x + w*2//3 - 2, arm2_y - 8, w//3 + 2, 7), border_radius=2)
    pygame.draw.rect(surf, color, (x + w - 7, arm2_y - 18, 7, 12), border_radius=2)


def draw_graph(surf: pygame.Surface, rect, history_best, history_avg, font_small):
    """Mini gráfico de evolución del fitness."""
    gx, gy, gw, gh = rect
    pygame.draw.rect(surf, BG, (gx, gy, gw, gh), border_radius=4)
    pygame.draw.rect(surf, GROUND_CLR, (gx, gy, gw, gh), 1, border_radius=4)

    if len(history_best) < 2:
        return

    max_val = max(history_best) if history_best else 1
    if max_val == 0:
        max_val = 1

    def to_px(i, val):
        px = gx + int(i / (len(history_best) - 1) * (gw - 4)) + 2
        py = gy + gh - 4 - int(val / max_val * (gh - 8))
        return px, py

    # Línea promedio
    if len(history_avg) >= 2:
        pts_avg = [to_px(i, v) for i, v in enumerate(history_avg)]
        pygame.draw.lines(surf, GRAPH_AVG, False, pts_avg, 1)

    # Línea mejor
    pts_best = [to_px(i, v) for i, v in enumerate(history_best)]
    pygame.draw.lines(surf, GRAPH_LINE, False, pts_best, 2)

    lbl = font_small.render("FITNESS", True, DIM_TEXT)
    surf.blit(lbl, (gx + 4, gy + 2))


class GameVisualizer:
    def __init__(self, ga: GeneticAlgorithm, speed_multiplier: int = 1):
        pygame.init()
        self.screen = pygame.Surface((SCREEN_W + PANEL_W, SCREEN_H))
        self.window = pygame.display.set_mode((SCREEN_W + PANEL_W, SCREEN_H))
        pygame.display.set_caption("EvoSurvivor — Algoritmo Genético")

        self.font_big   = pygame.font.SysFont("monospace", 28, bold=True)
        self.font_mid   = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 12)
        self.clock = pygame.time.Clock()

        self.ga = ga
        self.speed_multiplier = speed_multiplier  # Frames por tick visual

        self.ground_scroll = 0
        self.running = True

    def run(self):
        """Bucle principal: generación tras generación."""
        while self.running:
            self._run_generation()
            if not self.running:
                break
            # Evolucionar
            self.ga.evolve()

        pygame.quit()

    def _run_generation(self):
        """Simula y visualiza una generación completa."""
        population = self.ga.population
        nets = [NeuralNetwork(ind.weights) for ind in population]
        env = GameEnvironment(len(population))

        best_idx = 0  # Índice del mejor dino (para resaltarlo)

        while True:
            # Eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.speed_multiplier = min(self.speed_multiplier + 1, 10)
                    if event.key == pygame.K_DOWN:
                        self.speed_multiplier = max(self.speed_multiplier - 1, 1)
                    if event.key == pygame.K_s:
                        self.ga.save("best_dino.json")

            # Simular múltiples frames por tick visual
            any_alive = True
            for _ in range(self.speed_multiplier):
                any_alive = env.step(nets)
                if not any_alive:
                    break

            # Actualizar fitness en el AG
            scores = env.get_fitness_scores()
            for i, ind in enumerate(population):
                ind.fitness = scores[i]
                ind.alive = env.dinos[i].alive

            # Mejor dino vivo
            alive_indices = [i for i, d in enumerate(env.dinos) if d.alive]
            if alive_indices:
                best_idx = max(alive_indices, key=lambda i: scores[i])

            self._render(env, best_idx)

            if not any_alive:
                break

    def _render(self, env: GameEnvironment, best_idx: int):
        self.screen.fill(BG)

        # ── Área del juego ──
        game_area = pygame.Surface((SCREEN_W, SCREEN_H))
        game_area.fill(BG)

        # Suelo animado
        self.ground_scroll = (self.ground_scroll + env.speed) % 60
        for gx in range(-60, SCREEN_W + 60, 60):
            rx = int(gx - self.ground_scroll)
            pygame.draw.rect(game_area, GROUND_CLR, (rx, GROUND_Y + 48, 30, 3))

        pygame.draw.line(game_area, GROUND_CLR, (0, GROUND_Y + 48), (SCREEN_W, GROUND_Y + 48), 2)

        # Cactus
        for cactus in env.cacti:
            draw_cactus(game_area, cactus)

        # Dinos (muertos primero, luego vivos)
        dinos = env.dinos
        for i, dino in enumerate(dinos):
            if not dino.alive:
                draw_dino(game_area, dino, DINO_DEAD)
        for i, dino in enumerate(dinos):
            if dino.alive and i != best_idx:
                draw_dino(game_area, dino, DINO_ALIVE)
        # Mejor en amarillo
        if 0 <= best_idx < len(dinos) and dinos[best_idx].alive:
            draw_dino(game_area, dinos[best_idx], DINO_BEST)

        # HUD
        alive_txt = self.font_mid.render(
            f"Vivos: {env.alive_count}/{len(dinos)}", True, DINO_ALIVE)
        game_area.blit(alive_txt, (10, 10))
        speed_txt = self.font_small.render(
            f"Velocidad: {env.speed:.1f}  |  ↑↓ cambiar rapidez  [S] guardar",
            True, DIM_TEXT)
        game_area.blit(speed_txt, (10, SCREEN_H - 20))

        self.screen.blit(game_area, (0, 0))

        # ── Panel lateral ──
        self._draw_panel()

        self.window.blit(self.screen, (0, 0))
        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw_panel(self):
        px = SCREEN_W
        panel = pygame.Surface((PANEL_W, SCREEN_H))
        panel.fill(BG2)
        pygame.draw.line(panel, GROUND_CLR, (0, 0), (0, SCREEN_H), 2)

        stats = self.ga.get_stats()
        y = 16

        def label(text, color=TEXT_CLR, font=None):
            nonlocal y
            f = font or self.font_mid
            surf = f.render(text, True, color)
            panel.blit(surf, (12, y))
            y += surf.get_height() + 4

        def spacer(h=8):
            nonlocal y
            y += h

        label("EvoSurvivor", DINO_BEST, self.font_big)
        spacer()
        pygame.draw.line(panel, GROUND_CLR, (8, y), (PANEL_W - 8, y))
        spacer(6)

        label(f"Generación   {stats['generation']}", ACCENT)
        label(f"Vivos        {stats['alive']}", DINO_ALIVE)
        spacer()

        label("FITNESS", DIM_TEXT, self.font_small)
        label(f"Mejor actual  {stats['best_current']:.0f}")
        label(f"Promedio      {stats['avg_current']:.0f}")
        label(f"Mejor total   {stats['best_ever']:.0f}", DINO_BEST)
        spacer()

        label("PARÁMETROS", DIM_TEXT, self.font_small)
        label(f"Población     {self.ga.population_size}", TEXT_CLR, self.font_small)
        label(f"Mutación      {self.ga.mutation_rate:.0%}", TEXT_CLR, self.font_small)
        label(f"Cruce         {self.ga.crossover_rate:.0%}", TEXT_CLR, self.font_small)
        label(f"Elitismo      {self.ga.elitism}", TEXT_CLR, self.font_small)
        spacer()

        # Gráfico
        graph_rect = (8, y, PANEL_W - 16, 130)
        draw_graph(panel, graph_rect,
                   stats["history_best"], stats["history_avg"], self.font_small)
        y += 135

        label("━ Mejor   ━ Promedio", DIM_TEXT, self.font_small)
        spacer()
        label(f"x{self.speed_multiplier} velocidad", ACCENT, self.font_small)

        self.screen.blit(panel, (px, 0))