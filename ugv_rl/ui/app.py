import pygame
import numpy as np
import sys
import yaml
import json
import time

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0) # Goal
GREEN = (0, 255, 0) # Start
BLUE = (0, 0, 255) # Robot

class UGVApp:
    def __init__(self, config_path='config.yaml'):
        pygame.init()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.grid_size = self.config['env'].get('grid_size', 10)
        self.cell_size_m = self.config['env'].get('cell_size', 1.0)
        
        # Window settings
        self.cell_px = 60 # Pixels per cell
        self.width = self.grid_size * self.cell_px
        self.height = self.grid_size * self.cell_px + 100 # Extra space for controls
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("UGV RL Map Editor & Visualizer")
        
        # Map Data
        self.map_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.start_pos = [0, 0] # Grid indices
        self.goal_pos = [self.grid_size-1, self.grid_size-1]
        
        # Mode: 'edit', 'run'
        self.mode = 'edit'
        self.font = pygame.font.SysFont(None, 24)
        
        # Robot State (for visualization)
        self.robot_pose = None # (x, y, theta)

    def draw_grid(self):
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x*self.cell_px, y*self.cell_px, self.cell_px, self.cell_px)
                
                color = WHITE
                if self.map_grid[x, y] == 1:
                    color = BLACK
                elif [x, y] == self.start_pos:
                    color = GREEN
                elif [x, y] == self.goal_pos:
                    color = RED
                    
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, GRAY, rect, 1) # Grid lines

    def draw_robot(self):
        if self.robot_pose is None:
            return
            
        x_m, y_m, theta = self.robot_pose
        
        # Convert meters to pixels
        # Assuming map starts at (0,0)m top-left? No, usually coordinate systems...
        # Let's map simulation (x right, y down/up?) to screen (x right, y down)
        # Using simple scaling: px = m / cell_size_m * cell_px
        
        cx = int(x_m / self.cell_size_m * self.cell_px)
        cy = int(y_m / self.cell_size_m * self.cell_px)
        
        # Draw body
        pygame.draw.circle(self.screen, BLUE, (cx, cy), int(self.cell_px/3))
        
        # Draw heading
        line_len = int(self.cell_px/2)
        ex = cx + int(math.cos(theta) * line_len)
        ey = cy + int(math.sin(theta) * line_len)
        pygame.draw.line(self.screen, BLACK, (cx, cy), (ex, ey), 3)

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
            
            # UI Controls Text
            txt = f"Mode: {self.mode.upper()} | Click: Toggle Wall | S: Save | L: Load | Q: Quit"
            img = self.font.render(txt, True, BLACK)
            self.screen.blit(img, (10, self.height - 80))
            
            txt2 = f"Press 1: Set Start | Press 2: Set Goal"
            img2 = self.font.render(txt2, True, BLACK)
            self.screen.blit(img2, (10, self.height - 50))

            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    # Check if click is within grid
                    if my < self.grid_size * self.cell_px:
                        gx = mx // self.cell_px
                        gy = my // self.cell_px
                        
                        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                            # Left Click combined with key mods or mode?
                            # Let's use keys to set mode or just click to toggle wall
                            keys = pygame.key.get_pressed()
                            
                            if keys[pygame.K_1]: # Set Start
                                self.start_pos = [gx, gy]
                                # Clear other stuff if needed
                            elif keys[pygame.K_2]: # Set Goal
                                self.goal_pos = [gx, gy]
                            else:
                                # Toggle Wall
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
    import math
    app = UGVApp()
    app.run()
