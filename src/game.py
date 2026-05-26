"""
Pygame left/right game controlled by EEG motor imagery predictions.

Can run in two modes:
    - BCI mode: wired to realtime.py pipeline
    - Demo mode: keyboard control (for testing without a model)

Usage:
    python src/game.py --demo                    # keyboard control
    python src/game.py --model models/best_eegnet.pt --board synthetic
    python src/game.py --model models/best_eegnet.pt --board cyton --port COM3
"""

import argparse
import queue
import threading
import time
import random

import pygame

LABEL_LEFT = 0
LABEL_RIGHT = 1
LABEL_NONE = -1

# ── game constants ─────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 900, 600
FPS = 60
PLAYER_SPEED = 5
PLAYER_W, PLAYER_H = 60, 20
OBSTACLE_W, OBSTACLE_H = 80, 20
OBSTACLE_SPEED_INIT = 3
OBSTACLE_SPEED_MAX = 8
OBSTACLE_INTERVAL_MS = 1800

COLORS = {
    "bg": (20, 20, 35),
    "player": (80, 200, 120),
    "obstacle": (220, 80, 80),
    "text": (240, 240, 240),
    "gray": (120, 120, 120),
    "hud_left": (70, 130, 220),
    "hud_right": (220, 90, 70),
    "hud_none": (80, 80, 80),
}


class Player:
    def __init__(self):
        self.x = SCREEN_W // 2 - PLAYER_W // 2
        self.y = SCREEN_H - 60
        self.rect = pygame.Rect(self.x, self.y, PLAYER_W, PLAYER_H)

    def move(self, direction: int):
        """direction: -1=left, 0=none, 1=right"""
        self.rect.x += direction * PLAYER_SPEED
        self.rect.x = max(0, min(SCREEN_W - PLAYER_W, self.rect.x))

    def draw(self, screen):
        pygame.draw.rect(screen, COLORS["player"], self.rect, border_radius=6)


class Obstacle:
    def __init__(self, speed: float):
        self.rect = pygame.Rect(
            random.randint(0, SCREEN_W - OBSTACLE_W), -OBSTACLE_H,
            OBSTACLE_W, OBSTACLE_H
        )
        self.speed = speed

    def update(self) -> bool:
        """Returns True if obstacle went off screen."""
        self.rect.y += self.speed
        return self.rect.y > SCREEN_H

    def draw(self, screen):
        pygame.draw.rect(screen, COLORS["obstacle"], self.rect, border_radius=4)


def draw_hud(screen, font_sm, score: int, action: int, confidence: float = 0.0):
    # score
    score_surf = font_sm.render(f"Score: {score}", True, COLORS["text"])
    screen.blit(score_surf, (16, 16))

    # prediction indicator
    labels = {LABEL_LEFT: ("← LEFT", "hud_left"),
               LABEL_RIGHT: ("RIGHT →", "hud_right"),
               LABEL_NONE: ("  ---  ", "hud_none")}
    label_text, color_key = labels[action]
    pred_surf = font_sm.render(f"Prediction: {label_text}", True, COLORS[color_key])
    screen.blit(pred_surf, (SCREEN_W - pred_surf.get_width() - 16, 16))


def run_game(action_queue: queue.Queue, demo: bool = False):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("EEG Motor Imagery — Game")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 64, bold=True)
    font_sm = pygame.font.SysFont("Arial", 28)

    player = Player()
    obstacles = []
    score = 0
    current_action = LABEL_NONE
    obstacle_speed = OBSTACLE_SPEED_INIT
    last_obstacle_time = pygame.time.get_ticks()
    game_over = False

    while True:
        clock.tick(FPS)

        # ── events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                if game_over and event.key == pygame.K_r:
                    run_game(action_queue, demo)   # restart
                    return

        if game_over:
            screen.fill(COLORS["bg"])
            go_surf = font.render("GAME OVER", True, COLORS["obstacle"])
            sc_surf = font_sm.render(f"Score: {score}   (R to restart, ESC to quit)",
                                     True, COLORS["text"])
            screen.blit(go_surf, go_surf.get_rect(center=(SCREEN_W//2, SCREEN_H//2 - 40)))
            screen.blit(sc_surf, sc_surf.get_rect(center=(SCREEN_W//2, SCREEN_H//2 + 40)))
            pygame.display.flip()
            continue

        # ── get action ────────────────────────────────────────────────────────
        if demo:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                current_action = LABEL_LEFT
            elif keys[pygame.K_RIGHT]:
                current_action = LABEL_RIGHT
            else:
                current_action = LABEL_NONE
        else:
            try:
                current_action = action_queue.get_nowait()
            except queue.Empty:
                pass   # keep last action

        # ── update player ─────────────────────────────────────────────────────
        direction = {LABEL_LEFT: -1, LABEL_RIGHT: 1, LABEL_NONE: 0}[current_action]
        player.move(direction)

        # ── spawn obstacles ───────────────────────────────────────────────────
        now = pygame.time.get_ticks()
        if now - last_obstacle_time > OBSTACLE_INTERVAL_MS:
            obstacles.append(Obstacle(obstacle_speed))
            last_obstacle_time = now
            score += 1
            obstacle_speed = min(obstacle_speed + 0.1, OBSTACLE_SPEED_MAX)

        # ── update obstacles ──────────────────────────────────────────────────
        obstacles = [o for o in obstacles if not o.update()]

        # ── collision ─────────────────────────────────────────────────────────
        for o in obstacles:
            if player.rect.colliderect(o.rect):
                game_over = True

        # ── draw ──────────────────────────────────────────────────────────────
        screen.fill(COLORS["bg"])
        player.draw(screen)
        for o in obstacles:
            o.draw(screen)
        draw_hud(screen, font_sm, score, current_action)
        pygame.display.flip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Keyboard control mode")
    parser.add_argument("--model", default="models/best_eegnet.pt")
    parser.add_argument("--stats", default="data/processed/standardizer_realtime.npz")
    parser.add_argument("--board", default="synthetic", choices=["cyton", "synthetic"])
    parser.add_argument("--port", default="COM3")
    args = parser.parse_args()

    if args.demo:
        run_game(queue.Queue(), demo=True)
    else:
        from src.realtime import build_pipeline, run_pipeline
        model, standardizer, board, eeg_channels = build_pipeline(
            args.model, args.stats, args.board, args.port
        )
        action_queue, stop_event = run_pipeline(model, standardizer, board, eeg_channels)
        try:
            run_game(action_queue, demo=False)
        finally:
            stop_event.set()
            board.release_session()


if __name__ == "__main__":
    main()
