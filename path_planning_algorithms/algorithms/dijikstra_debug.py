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
        self.step_delay = 1         # Reduced to show one step at a time
        self.step_counter = 0
        self.show_decision_process = True  # Toggle for showing/hiding decision process
        self.step_by_step_mode = False     # Step-by-step mode
        self.step_history = []             # History of algorithm steps for visualization
        self.current_step_index = -1       # Current step being displayed
        
        # Colors for visualization
        self.colors = {
            'start': (0, 200, 0),       # Green
            'goal': (200, 0, 0),        # Red
            'path': (255, 255, 0),      # Yellow
            'visited': (100, 100, 255, 100),  # Light blue with transparency
            'frontier': (100, 255, 100, 100),  # Light green with transparency
            'arrow': (200, 200, 200),    # Light gray for arrows
            'current_node': (255, 165, 0, 200),  # Orange for current node being processed
            'considered': (255, 255, 0, 100)     # Yellow for neighbors being considered
        }
        
    def reset(self):
        """Reset the path planning state."""
        self.path = []
        self.visited_cells = set()
        self.frontier = set()
        self.found_path = False
        self.is_planning = False
        self.step_history = []
        self.current_step_index = -1
        
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
        
        # Initialize algorithm for step-by-step execution
        self.pq = [(0, self.start_cell)]
        self.cost_so_far = {self.start_cell: 0}
        self.came_from = {self.start_cell: None}
        self.frontier.add(self.start_cell)
        
        # Create first step in history
        self.step_history.append({
            'visited': set(),
            'frontier': {self.start_cell},
            'current_node': self.start_cell,
            'considered': set(),
            'cost_so_far': {self.start_cell: 0},
            'came_from': {self.start_cell: None},
            'description': "Starting at source node"
        })
        self.current_step_index = 0
        
        return True
        
    def toggle_step_by_step_mode(self):
        """Toggle step-by-step mode."""
        self.step_by_step_mode = not self.step_by_step_mode
        
    def next_step(self):
        """Move to the next step in step-by-step mode."""
        if self.step_by_step_mode and self.current_step_index < len(self.step_history) - 1:
            self.current_step_index += 1
            return True
        elif self.step_by_step_mode and self.is_planning:
            # Calculate a new step
            self.dijkstra_single_step()
            return True
        return False
        
    def prev_step(self):
        """Move to the previous step in step-by-step mode."""
        if self.step_by_step_mode and self.current_step_index > 0:
            self.current_step_index -= 1
            return True
        return False
        
    def dijkstra_single_step(self):
        """Execute a single step of Dijkstra's algorithm for step-by-step visualization."""
        if not self.is_planning or self.found_path:
            return
        
        if not self.pq:
            self.is_planning = False
            self.step_history.append({
                'visited': self.visited_cells.copy(),
                'frontier': set(),
                'current_node': None,
                'considered': set(),
                'cost_so_far': self.cost_so_far.copy(),
                'came_from': self.came_from.copy(),
                'description': "No path found!"
            })
            self.current_step_index = len(self.step_history) - 1
            print("No path found!")
            return
        
        # Get the cell with the lowest cost
        current_cost, current_cell = heapq.heappop(self.pq)
        
        # Remove from frontier
        if current_cell in self.frontier:
            self.frontier.remove(current_cell)
        
        # If we've already visited this cell, skip it
        if current_cell in self.visited_cells:
            self.dijkstra_single_step()  # Try next node
            return
        
        # Add to visited
        self.visited_cells.add(current_cell)
        
        # Check if we've reached the goal
        if current_cell == self.goal_cell:
            self.reconstruct_path()
            self.found_path = True
            self.is_planning = False
            
            self.step_history.append({
                'visited': self.visited_cells.copy(),
                'frontier': self.frontier.copy(),
                'current_node': current_cell,
                'considered': set(),
                'cost_so_far': self.cost_so_far.copy(),
                'came_from': self.came_from.copy(),
                'path': self.path,
                'description': "Goal reached! Path found."
            })
            self.current_step_index = len(self.step_history) - 1
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
        
        # Track which neighbors we're considering in this step
        considered_neighbors = set()
        updated_neighbors = set()
        
        for next_cell in neighbors:
            # Check if valid cell and not occupied
            if (0 <= next_cell[0] < self.grid.rows and 
                0 <= next_cell[1] < self.grid.cols and 
                not self.grid.is_cell_occupied(*next_cell) and
                next_cell not in self.visited_cells):
                
                considered_neighbors.add(next_cell)
                
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
                    updated_neighbors.add(next_cell)
        
        # Record this step in history
        self.step_history.append({
            'visited': self.visited_cells.copy(),
            'frontier': self.frontier.copy(),
            'current_node': current_cell,
            'considered': considered_neighbors,
            'updated': updated_neighbors,
            'cost_so_far': self.cost_so_far.copy(),
            'came_from': self.came_from.copy(),
            'description': f"Processing node at {current_cell} with cost {current_cost:.1f}"
        })
        self.current_step_index = len(self.step_history) - 1
        
    def dijkstra_step(self):
        """Execute steps of Dijkstra's algorithm."""
        if self.step_by_step_mode:
            # In step-by-step mode, do nothing automatically
            return
        
        if not self.is_planning or self.found_path:
            return
            
        # Execute multiple steps based on step_delay
        for _ in range(self.step_delay):
            self.dijkstra_single_step()
            if not self.is_planning or self.found_path:
                break
    
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
        
        # If in step-by-step mode, draw the current step from history
        if self.step_by_step_mode and self.current_step_index >= 0 and self.current_step_index < len(self.step_history):
            step = self.step_history[self.current_step_index]
            
            # Draw step description at the top of the screen
            description_font = pygame.font.SysFont('Arial', 16)
            desc_surf = description_font.render(step['description'], True, (255, 255, 255))
            screen.blit(desc_surf, (10, 10))
            
            # Draw step number
            step_text = f"Step: {self.current_step_index + 1}/{len(self.step_history)}"
            step_surf = description_font.render(step_text, True, (255, 255, 255))
            screen.blit(step_surf, (10, 30))
            
            # Draw visited cells
            for row, col in step['visited']:
                rect = pygame.Rect(
                    col * cell_size,
                    row * cell_size,
                    cell_size,
                    cell_size
                )
                s = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                s.fill(self.colors['visited'])
                screen.blit(s, rect)
            
            # Draw frontier cells
            for row, col in step['frontier']:
                rect = pygame.Rect(
                    col * cell_size,
                    row * cell_size,
                    cell_size,
                    cell_size
                )
                s = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                s.fill(self.colors['frontier'])
                screen.blit(s, rect)
            
            # Draw current node being processed
            if step['current_node']:
                row, col = step['current_node']
                rect = pygame.Rect(
                    col * cell_size,
                    row * cell_size,
                    cell_size,
                    cell_size
                )
                s = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                s.fill(self.colors['current_node'])
                screen.blit(s, rect)
            
            # Draw considered neighbors
            if 'considered' in step:
                for row, col in step['considered']:
                    rect = pygame.Rect(
                        col * cell_size,
                        row * cell_size,
                        cell_size,
                        cell_size
                    )
                    s = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                    s.fill(self.colors['considered'])
                    screen.blit(s, rect)
            
            # Draw arrows and costs if show_decision_process is enabled
            if self.show_decision_process:
                for cell in step['visited']:
                    if cell in step['came_from'] and step['came_from'][cell] is not None:
                        # Draw direction arrow
                        curr_row, curr_col = cell
                        prev_row, prev_col = step['came_from'][cell]
                        
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
                
                # Draw cost values for all cells with costs
                for cell, cost in step['cost_so_far'].items():
                    row, col = cell
                    cost_text = f"{cost:.1f}"
                    cost_surf = font.render(cost_text, True, (255, 255, 255))
                    cost_rect = cost_surf.get_rect(center=(
                        col * cell_size + cell_size // 2,
                        row * cell_size + cell_size // 2
                    ))
                    screen.blit(cost_surf, cost_rect)
            
            # Draw path if found in this step
            if 'path' in step and step['path']:
                for i in range(1, len(step['path'])):
                    prev_row, prev_col = step['path'][i-1]
                    curr_row, curr_col = step['path'][i]
                    start_pos = (prev_col * cell_size + cell_size // 2, prev_row * cell_size + cell_size // 2)
                    end_pos = (curr_col * cell_size + cell_size // 2, curr_row * cell_size + cell_size // 2)
                    pygame.draw.line(screen, self.colors['path'], start_pos, end_pos, 4)
                
                # Highlight path nodes
                for i, (row, col) in enumerate(step['path']):
                    if (row, col) == self.start_cell or (row, col) == self.goal_cell:
                        continue
                    
                    center_pos = (col * cell_size + cell_size // 2, row * cell_size + cell_size // 2)
                    pygame.draw.circle(screen, (255, 200, 0), center_pos, cell_size // 4)
                    
                    # Add step number
                    step_text = str(i)
                    step_surf = font.render(step_text, True, (0, 0, 0))
                    step_rect = step_surf.get_rect(center=center_pos)
                    screen.blit(step_surf, step_rect)
        
        # If not in step-by-step mode, draw regular visualization
        else:
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
                
                # Then draw arrows and costs if show_decision_process is enabled
                if self.show_decision_process:
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
                if self.show_decision_process and hasattr(self, 'cost_so_far') and (row, col) in self.cost_so_far:
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
        
        # Always draw start and goal
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