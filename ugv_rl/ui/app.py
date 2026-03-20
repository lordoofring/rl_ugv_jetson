import pygame
import numpy as np
import sys
import yaml
import json
import math
import time

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)       # Goal
GREEN = (0, 255, 0)     # Start
BLUE = (0, 0, 255)      # Robot
LIGHT_BLUE = (173, 216, 230)  # Trail
YELLOW = (255, 255, 0)  # Action arrow
DARK_GREEN = (0, 150, 0)

INFO_BAR_HEIGHT = 120

class UGVApp:
    def __init__(self, config_path='config.yaml'):
        pygame.init()

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.grid_size = self.config['env'].get('grid_size', 10)
        self.cell_size_m = self.config['env'].get('cell_size', 1.0)

        self.cell_px = self.config['env'].get('pixels_per_cell', 60)
        self.width = self.grid_size * self.cell_px
        self.height = self.grid_size * self.cell_px + INFO_BAR_HEIGHT

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("UGV RL Map Editor & Visualizer")

        # Map Data
        self.map_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.start_pos = [0, 0]
        self.goal_pos = [self.grid_size - 1, self.grid_size - 1]

        self.mode = 'edit'
        self.font = pygame.font.SysFont(None, 24)
        self.font_big = pygame.font.SysFont(None, 32)

        # Robot state for visualization
        self.robot_pose = None  # (x, y, theta) in meters
        self.agent_cell = None  # (gx, gy) grid indices
        self.trail = []         # list of (gx, gy) visited cells
        self.action_name = ''
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_num = 0

    def draw_grid(self):
        for gx in range(self.grid_size):
            for gy in range(self.grid_size):
                rect = pygame.Rect(gx * self.cell_px, gy * self.cell_px, self.cell_px, self.cell_px)

                color = WHITE
                if self.map_grid[gx, gy] == 1:
                    color = BLACK
                elif [gx, gy] == self.start_pos:
                    color = GREEN
                elif [gx, gy] == self.goal_pos:
                    color = RED

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, GRAY, rect, 1)

    def draw_trail(self):
        for (gx, gy) in self.trail:
            rect = pygame.Rect(gx * self.cell_px + 2, gy * self.cell_px + 2,
                               self.cell_px - 4, self.cell_px - 4)
            trail_surface = pygame.Surface((self.cell_px - 4, self.cell_px - 4), pygame.SRCALPHA)
            trail_surface.fill((173, 216, 230, 100))  # semi-transparent light blue
            self.screen.blit(trail_surface, rect.topleft)

    def draw_agent(self):
        """Draw the agent at its grid cell position."""
        if self.agent_cell is None:
            return

        gx, gy = self.agent_cell
        cx = gx * self.cell_px + self.cell_px // 2
        cy = gy * self.cell_px + self.cell_px // 2
        radius = self.cell_px // 3

        # Body
        pygame.draw.circle(self.screen, BLUE, (cx, cy), radius)

        # Heading arrow
        if self.robot_pose and len(self.robot_pose) >= 3:
            theta = self.robot_pose[2]
            arrow_len = self.cell_px // 2
            ex = cx + int(math.cos(theta) * arrow_len)
            ey = cy + int(math.sin(theta) * arrow_len)
            pygame.draw.line(self.screen, YELLOW, (cx, cy), (ex, ey), 3)
            # Arrowhead
            head_size = 6
            left_x = ex - int(math.cos(theta - 0.5) * head_size)
            left_y = ey - int(math.sin(theta - 0.5) * head_size)
            right_x = ex - int(math.cos(theta + 0.5) * head_size)
            right_y = ey - int(math.sin(theta + 0.5) * head_size)
            pygame.draw.polygon(self.screen, YELLOW, [(ex, ey), (left_x, left_y), (right_x, right_y)])

    def draw_robot(self):
        """Legacy: draw robot from continuous meter position."""
        if self.robot_pose is None:
            return

        x_m, y_m, theta = self.robot_pose
        cx = int(x_m / self.cell_size_m * self.cell_px)
        cy = int(y_m / self.cell_size_m * self.cell_px)

        pygame.draw.circle(self.screen, BLUE, (cx, cy), int(self.cell_px / 3))

        line_len = int(self.cell_px / 2)
        ex = cx + int(math.cos(theta) * line_len)
        ey = cy + int(math.sin(theta) * line_len)
        pygame.draw.line(self.screen, BLACK, (cx, cy), (ex, ey), 3)

    def draw_info_bar(self):
        """Draw status info below the grid."""
        bar_y = self.grid_size * self.cell_px
        bar_rect = pygame.Rect(0, bar_y, self.width, INFO_BAR_HEIGHT)
        pygame.draw.rect(self.screen, (40, 40, 40), bar_rect)

        lines = [
            f"Episode: {self.episode_num}  |  Step: {self.step_count}  |  Reward: {self.episode_reward:.1f}",
            f"Action: {self.action_name}  |  Cell: {self.agent_cell}",
            "WASD to move  |  R=Reset  |  Q=Quit",
        ]

        if self.mode == 'edit':
            lines = [
                f"Mode: EDIT  |  Click: Toggle Wall  |  S: Save  |  L: Load  |  Q: Quit",
                f"Hold 1+Click: Set Start  |  Hold 2+Click: Set Goal",
            ]

        for i, line in enumerate(lines):
            img = self.font.render(line, True, WHITE)
            self.screen.blit(img, (10, bar_y + 10 + i * 28))

    def draw_goal_marker(self):
        """Draw a prominent goal marker."""
        gx, gy = self.goal_pos
        cx = gx * self.cell_px + self.cell_px // 2
        cy = gy * self.cell_px + self.cell_px // 2
        # Draw star/target
        pygame.draw.circle(self.screen, RED, (cx, cy), self.cell_px // 4, 3)
        pygame.draw.circle(self.screen, RED, (cx, cy), self.cell_px // 8)

    def draw_start_marker(self):
        """Draw a prominent start marker."""
        gx, gy = self.start_pos
        cx = gx * self.cell_px + self.cell_px // 2
        cy = gy * self.cell_px + self.cell_px // 2
        pygame.draw.circle(self.screen, DARK_GREEN, (cx, cy), self.cell_px // 4, 3)
        label = self.font.render("S", True, DARK_GREEN)
        self.screen.blit(label, (cx - 5, cy - 8))

    def render_run_frame(self):
        """Full render for run/test mode."""
        self.screen.fill(GRAY)
        self.draw_grid()
        self.draw_trail()
        self.draw_start_marker()
        self.draw_goal_marker()
        self.draw_agent()
        self.draw_info_bar()
        pygame.display.flip()

    def save_map(self, filename='map_data.json'):
        data = {
            "grid": self.map_grid.tolist(),
            "start": self.start_pos,
            "goal": self.goal_pos
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Map saved to {filename}")

    def load_map(self, filename='map_data.json'):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.map_grid = np.array(data['grid'])
                self.start_pos = data['start']
                self.goal_pos = data['goal']
            print(f"Map loaded from {filename}")
        except FileNotFoundError:
            print("Map file not found.")

    def run(self):
        clock = pygame.time.Clock()
        running = True

        while running:
            self.screen.fill(GRAY)
            self.draw_grid()
            self.draw_robot()
            self.draw_info_bar()

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    if my < self.grid_size * self.cell_px:
                        gx = mx // self.cell_px
                        gy = my // self.cell_px

                        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                            keys = pygame.key.get_pressed()
                            if keys[pygame.K_1]:
                                self.start_pos = [gx, gy]
                            elif keys[pygame.K_2]:
                                self.goal_pos = [gx, gy]
                            else:
                                if [gx, gy] != self.start_pos and [gx, gy] != self.goal_pos:
                                    self.map_grid[gx, gy] = 1 - self.map_grid[gx, gy]

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_s:
                        self.save_map('map.json')
                    elif event.key == pygame.K_l:
                        self.load_map('map.json')

            clock.tick(30)

        pygame.quit()

if __name__ == '__main__':
    app = UGVApp()
    app.run()
