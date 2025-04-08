import pygame
import heapq
from collections import defaultdict

class PathFinder:
    """
    A class that implements Dijkstra's algorithm for path planning.
    Visualizes the search process and final path.
    """
    def __init__(self, grid):
        self.grid = grid
        self.path = []
        self.visited_cells = set()  # For visualization
        self.frontier = set()       # For visualization
        self.start_cell = None
        self.goal_cell = None
        self.found_path = False
        self.is_planning = False
        self.step_delay = 5         # Steps per frame
        self.step_counter = 0
        self.show_decision_process = True  # Toggle for showing/hiding decision process
        
        # Colors for visualization
        self.colors = {
            'start': (0, 200, 0),       # Green
            'goal': (200, 0, 0),        # Red
            'path': (255, 255, 0),      # Yellow
            'visited': (100, 100, 255, 100),  # Light blue with transparency
            'frontier': (100, 255, 100, 100),  # Light green with transparency
            'arrow': (200, 200, 200)    # Light gray for arrows
        }
        
    def reset(self):
        """Reset the path planning state."""
        self.path = []
        self.visited_cells = set()
        self.frontier = set()
        self.found_path = False
        self.is_planning = False
        
    def set_start(self, x, y):
        """Set the start position for path planning."""
        cell = self.grid.get_cell(x, y)
        if cell and not self.grid.is_cell_occupied(*cell):
            self.start_cell = cell
            return True
        return False
        
    def set_goal(self, x, y):
        """Set the goal position for path planning."""
        cell = self.grid.get_cell(x, y)
        if cell and not self.grid.is_cell_occupied(*cell):
            self.goal_cell = cell
            return True
        return False
    
    def start_planning(self):
        """Begin the path planning process using Dijkstra's algorithm."""
        if not self.start_cell or not self.goal_cell:
            print("Start or goal not set")
            return False
            
        self.reset()
        self.is_planning = True
        return True
        
    def dijkstra_step(self):
        """Execute one step of Dijkstra's algorithm."""
        if not self.is_planning or self.found_path:
            return
            
        # Initialize if this is the first step
        if not self.frontier:
            # Priority queue: (cost, (row, col))
            self.pq = [(0, self.start_cell)]
            # Dictionary to store cost to reach each node
            self.cost_so_far = {self.start_cell: 0}
            # Dictionary to store the path
            self.came_from = {self.start_cell: None}
            # Add start to frontier for visualization
            self.frontier.add(self.start_cell)
            
        # Step through the algorithm
        for _ in range(self.step_delay):
            if not self.pq:
                self.is_planning = False
                print("No path found!")
                return
                
            # Get the cell with the lowest cost
            current_cost, current_cell = heapq.heappop(self.pq)
            
            # Remove from frontier (for visualization)
            if current_cell in self.frontier:
                self.frontier.remove(current_cell)
                
            # Add to visited (for visualization)
            self.visited_cells.add(current_cell)
            
            # Check if we've reached the goal
            if current_cell == self.goal_cell:
                self.reconstruct_path()
                self.found_path = True
                self.is_planning = False
                print("Path found!")
                return
                
            # Explore neighbors
            row, col = current_cell
            neighbors = [
                (row-1, col), (row+1, col),   # Up, Down
                (row, col-1), (row, col+1),   # Left, Right
                (row-1, col-1), (row-1, col+1),  # Diagonal: Up-Left, Up-Right
                (row+1, col-1), (row+1, col+1)   # Diagonal: Down-Left, Down-Right
            ]
            
            for next_cell in neighbors:
                # Check if valid cell and not occupied
                if (0 <= next_cell[0] < self.grid.rows and 
                    0 <= next_cell[1] < self.grid.cols and 
                    not self.grid.is_cell_occupied(*next_cell)):
                    
                    # Calculate new cost (1 for orthogonal, 1.414 for diagonal)
                    is_diagonal = (next_cell[0] != row and next_cell[1] != col)
                    movement_cost = 1.414 if is_diagonal else 1.0
                    new_cost = self.cost_so_far[current_cell] + movement_cost
                    
                    # If we haven't visited this cell or found a cheaper path
                    if next_cell not in self.cost_so_far or new_cost < self.cost_so_far[next_cell]:
                        self.cost_so_far[next_cell] = new_cost
                        heapq.heappush(self.pq, (new_cost, next_cell))
                        self.came_from[next_cell] = current_cell
                        
                        # Add to frontier for visualization
                        self.frontier.add(next_cell)
    
    def reconstruct_path(self):
        """Reconstruct the path from start to goal using the came_from dictionary."""
        if self.goal_cell in self.came_from:
            self.path = []
            current = self.goal_cell
            while current:
                self.path.append(current)
                current = self.came_from[current]
            self.path.reverse()
    
    def draw(self, screen):
        """Draw the path planning visualization including visited cells, frontier, and path."""
        cell_size = self.grid.cell_size
        font = pygame.font.SysFont('Arial', 10)  # Small font for cost values
        
        # Draw visited cells with cost and direction indicators
        if hasattr(self, 'came_from') and hasattr(self, 'cost_so_far'):
            # First draw all visited cells
            for row, col in self.visited_cells:
                rect = pygame.Rect(
                    col * cell_size,
                    row * cell_size,
                    cell_size,
                    cell_size
                )
                s = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                s.fill(self.colors['visited'])
                screen.blit(s, rect)
            
            # Then draw arrows and costs
            for cell in self.visited_cells:
                if cell in self.came_from and self.came_from[cell] is not None:
                    # Draw direction arrow
                    curr_row, curr_col = cell
                    prev_row, prev_col = self.came_from[cell]
                    
                    # Calculate arrow positions
                    start_x = prev_col * cell_size + cell_size // 2
                    start_y = prev_row * cell_size + cell_size // 2
                    end_x = curr_col * cell_size + cell_size // 2
                    end_y = curr_row * cell_size + cell_size // 2
                    
                    # Draw arrow
                    pygame.draw.line(screen, (255, 255, 255), (start_x, start_y), (end_x, end_y), 1)
                    
                    # Calculate arrowhead
                    dx = end_x - start_x
                    dy = end_y - start_y
                    length = max(1, (dx**2 + dy**2)**0.5)
                    dx, dy = dx/length, dy/length
                    
                    # Perpendicular vectors for arrowhead
                    px, py = -dy, dx
                    
                    # Draw arrowhead
                    arrow_size = cell_size // 4
                    pygame.draw.polygon(screen, (255, 255, 255), [
                        (end_x, end_y),
                        (end_x - arrow_size*dx + arrow_size*px//2, end_y - arrow_size*dy + arrow_size*py//2),
                        (end_x - arrow_size*dx - arrow_size*px//2, end_y - arrow_size*dy - arrow_size*py//2)
                    ])
                
                # Draw cost value
                if cell in self.cost_so_far:
                    cost = self.cost_so_far[cell]
                    cost_text = f"{cost:.1f}"
                    cost_surf = font.render(cost_text, True, (255, 255, 255))
                    cost_rect = cost_surf.get_rect(center=(
                        cell[1] * cell_size + cell_size // 2,
                        cell[0] * cell_size + cell_size // 2
                    ))
                    screen.blit(cost_surf, cost_rect)
        
        # Draw frontier cells
        for row, col in self.frontier:
            rect = pygame.Rect(
                col * cell_size,
                row * cell_size,
                cell_size,
                cell_size
            )
            s = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
            s.fill(self.colors['frontier'])
            screen.blit(s, rect)
            
            # Draw cost for frontier cells
            if hasattr(self, 'cost_so_far') and (row, col) in self.cost_so_far:
                cost = self.cost_so_far[(row, col)]
                cost_text = f"{cost:.1f}"
                cost_surf = font.render(cost_text, True, (0, 0, 0))
                cost_rect = cost_surf.get_rect(center=(
                    col * cell_size + cell_size // 2,
                    row * cell_size + cell_size // 2
                ))
                screen.blit(cost_surf, cost_rect)
        
        # Draw the final path with highlighted nodes and costs
        if self.path:
            # First draw the connecting lines
            for i in range(1, len(self.path)):
                prev_row, prev_col = self.path[i-1]
                curr_row, curr_col = self.path[i]
                start_pos = (prev_col * cell_size + cell_size // 2, prev_row * cell_size + cell_size // 2)
                end_pos = (curr_col * cell_size + cell_size // 2, curr_row * cell_size + cell_size // 2)
                pygame.draw.line(screen, self.colors['path'], start_pos, end_pos, 4)
            
            # Then highlight each node in the path
            for i, (row, col) in enumerate(self.path):
                # Skip start and goal which will be drawn separately
                if (row, col) == self.start_cell or (row, col) == self.goal_cell:
                    continue
                    
                # Draw a smaller circle for each path node
                center_pos = (col * cell_size + cell_size // 2, row * cell_size + cell_size // 2)
                pygame.draw.circle(screen, (255, 200, 0), center_pos, cell_size // 4)
                
                # Add step number to path
                step_text = str(i)
                step_surf = font.render(step_text, True, (0, 0, 0))
                step_rect = step_surf.get_rect(center=center_pos)
                screen.blit(step_surf, step_rect)
        
        # Draw start and goal
        if self.start_cell:
            row, col = self.start_cell
            center_pos = (col * cell_size + cell_size // 2, row * cell_size + cell_size // 2)
            pygame.draw.circle(screen, self.colors['start'], center_pos, cell_size // 2)
            
            # Add "S" label
            s_surf = font.render("S", True, (0, 0, 0))
            s_rect = s_surf.get_rect(center=center_pos)
            screen.blit(s_surf, s_rect)
            
        if self.goal_cell:
            row, col = self.goal_cell
            center_pos = (col * cell_size + cell_size // 2, row * cell_size + cell_size // 2)
            pygame.draw.circle(screen, self.colors['goal'], center_pos, cell_size // 2)
            
            # Add "G" label
            g_surf = font.render("G", True, (0, 0, 0))
            g_rect = g_surf.get_rect(center=center_pos)
            screen.blit(g_surf, g_rect)
    
    def get_path_stats(self):
        """Return statistics about the found path."""
        if not self.found_path or not self.path:
            return {
                'found': False,
                'length': 0,
                'cost': 0,
                'visited_count': len(self.visited_cells)
            }
        
        # Calculate path length and cost
        path_length = len(self.path) - 1  # Number of segments
        path_cost = self.cost_so_far[self.goal_cell] if self.goal_cell in self.cost_so_far else 0
        
        return {
            'found': True,
            'length': path_length,
            'cost': path_cost,
            'visited_count': len(self.visited_cells)
        }