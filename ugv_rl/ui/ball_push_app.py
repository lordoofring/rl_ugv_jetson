"""Pygame visualizer for the Ball Push environment."""

import pygame
import math

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (220, 40, 40)
BLUE = (40, 80, 220)
YELLOW = (255, 220, 0)
GREEN = (40, 180, 40)
DARK_GRAY = (60, 60, 60)

INFO_BAR_HEIGHT = 80


class BallPushApp:
    """Simple top-down visualizer for the ball push task."""

    def __init__(self, arena_size: float = 2.0, window_size: int = 500):
        pygame.init()
        self.arena_size = arena_size
        self.win_size = window_size
        self.scale = window_size / arena_size  # pixels per meter

        self.width = window_size
        self.height = window_size + INFO_BAR_HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Ball Push — UGV RL")
        self.font = pygame.font.SysFont(None, 22)

        self.robot_pos = (0, 0)
        self.robot_theta = 0.0
        self.ball_pos = (0, 0)
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_num = 0
        self.action_name = ""
        self.trail = []

    def world_to_screen(self, wx, wy):
        """Convert world coords (origin=center) to screen coords (origin=top-left)."""
        sx = int((wx + self.arena_size / 2) * self.scale)
        sy = int((-wy + self.arena_size / 2) * self.scale)  # flip Y
        return sx, sy

    def render(self):
        self.screen.fill(WHITE)

        # Arena boundary
        half = self.arena_size / 2
        tl = self.world_to_screen(-half, half)
        br = self.world_to_screen(half, -half)
        arena_rect = pygame.Rect(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])
        pygame.draw.rect(self.screen, GRAY, arena_rect)
        pygame.draw.rect(self.screen, BLACK, arena_rect, 3)

        # Trail
        for pos in self.trail:
            sx, sy = self.world_to_screen(pos[0], pos[1])
            pygame.draw.circle(self.screen, (180, 210, 240), (sx, sy), 3)

        # Ball
        bx, by = self.world_to_screen(self.ball_pos[0], self.ball_pos[1])
        pygame.draw.circle(self.screen, RED, (bx, by), max(8, int(0.05 * self.scale)))

        # Robot
        rx, ry = self.world_to_screen(self.robot_pos[0], self.robot_pos[1])
        pygame.draw.circle(self.screen, BLUE, (rx, ry), max(10, int(0.10 * self.scale)))

        # Heading arrow
        arrow_len = max(15, int(0.15 * self.scale))
        ex = rx + int(math.cos(self.robot_theta) * arrow_len)
        ey = ry - int(math.sin(self.robot_theta) * arrow_len)  # flip Y
        pygame.draw.line(self.screen, YELLOW, (rx, ry), (ex, ey), 3)

        # Info bar
        bar_y = self.win_size
        pygame.draw.rect(self.screen, DARK_GRAY, (0, bar_y, self.width, INFO_BAR_HEIGHT))
        lines = [
            f"Episode: {self.episode_num}  |  Step: {self.step_count}  |  Reward: {self.episode_reward:.1f}",
            f"Action: {self.action_name}  |  Ball: ({self.ball_pos[0]:.2f}, {self.ball_pos[1]:.2f})",
        ]
        for i, line in enumerate(lines):
            img = self.font.render(line, True, WHITE)
            self.screen.blit(img, (10, bar_y + 8 + i * 26))

        pygame.display.flip()
