import pygame
import heapq
import random
import math
from collections import defaultdict
from enum import Enum, auto

class Algorithm(Enum):
    DIJKSTRA = auto()
    ASTAR = auto()
    BFS = auto()
    DFS = auto()
    RRT = auto()
    POTENTIAL_FIELD = auto()

class PathFinder:
    """
    A class that implements multiple path planning algorithms.
    Supports Dijkstra, A*, BFS, and DFS with step-by-step visualization.
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
        self.step_delay = 5         # Steps per frame in automatic mode
        self.step_counter = 0
        self.show_decision_process = True  # Toggle for showing/hiding decision process
        
        # Step-by-step mode variables
        self.step_by_step_mode = False
        self.history = []           # Store snapshots for step navigation
        self.history_index = -1     # Current position in history
        
        # Current algorithm
        self.current_algorithm = Algorithm.ASTAR
        
        # Colors for visualization
        self.colors = {
            'start': (0, 200, 0),       # Green
            'goal': (200, 0, 0),        # Red
            'path': (255, 255, 0),      # Yellow
            'visited': (100, 100, 255, 100),  # Light blue with transparency
            'frontier': (100, 255, 100, 100),  # Light green with transparency
            'arrow': (200, 200, 200),   # Light gray for arrows
            'rrt_node': (255, 165, 0),  # Orange for RRT nodes
            'rrt_edge': (200, 200, 200), # Light gray for RRT edges
            'potential_field': (150, 100, 255, 100)  # Purple for potential field
        }

        # RRT-specific parameters
        self.rrt_nodes = []         # List of nodes in the RRT
        self.rrt_edges = []         # List of edges in the RRT
        self.rrt_step_size = 2      # Step size for RRT extension (in cells)
        self.rrt_goal_sample_rate = 0.5  # Probability of sampling goal
        self.rrt_max_iterations = 2000   # Maximum iterations for RRT
        
        # Potential Field parameters
        self.potential_field = None  # Grid of potential values
        self.attraction_factor = 10.0  # Strength of goal attraction
        self.repulsion_factor = 100.0  # Strength of obstacle repulsion
        self.repulsion_radius = 3     # Radius of obstacle influence (in cells)
        self.potential_step_size = 0.5 # Step size for gradient descent
        self.potential_max_iterations = 500  # Max iterations for potential field path
        
    def reset(self):
        """Reset the path planning state."""
        self.path = []
        self.visited_cells = set()
        self.frontier = set()
        self.found_path = False
        self.is_planning = False
        self.history = []
        self.history_index = -1
        
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
    
    def set_algorithm(self, algorithm):
        """Set the current path planning algorithm."""
        print(algorithm, self.current_algorithm)
        self.current_algorithm = algorithm
        
        self.reset()
        return True
    
    def toggle_step_by_step_mode(self):
        """Toggle between automatic and step-by-step planning modes."""
        self.step_by_step_mode = not self.step_by_step_mode
        print(f"Step-by-step mode: {'ON' if self.step_by_step_mode else 'OFF'}")
        
    def next_step(self):
        """Execute next step in step-by-step mode."""
        if self.step_by_step_mode and self.is_planning:
            self._execute_algorithm_step()
            # We'll save history after the step is executed
        elif self.step_by_step_mode and self.history:
            # Navigate forward in history if we have history
            if self.history_index < len(self.history) - 1:
                self.history_index += 1
                self._restore_state_from_history()
    
    def prev_step(self):
        """Go back one step in step-by-step mode."""
        if self.step_by_step_mode and self.history and self.history_index > 0:
            self.history_index -= 1
            self._restore_state_from_history()
    
    def _save_state_to_history(self):
        """Save the current state to history for step-by-step navigation."""
        # Create a snapshot of the current state
        state = {
            'visited_cells': self.visited_cells.copy(),
            'frontier': self.frontier.copy(),
            'path': self.path.copy() if self.path else [],
            'found_path': self.found_path,
            'is_planning': self.is_planning
        }
        
        # If we have cost_so_far and came_from, save them too
        if hasattr(self, 'cost_so_far'):
            state['cost_so_far'] = self.cost_so_far.copy()
        if hasattr(self, 'came_from'):
            state['came_from'] = self.came_from.copy()
        
        # If we've been navigating history and now make a new step,
        # truncate the future history
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        
        # Add the new state to history
        self.history.append(state)
        self.history_index = len(self.history) - 1
    
    def _restore_state_from_history(self):
        """Restore a state from history for step-by-step navigation."""
        if 0 <= self.history_index < len(self.history):
            state = self.history[self.history_index]
            
            # Restore the state
            self.visited_cells = state['visited_cells']
            self.frontier = state['frontier']
            self.path = state['path']
            self.found_path = state['found_path']
            self.is_planning = state['is_planning']
            
            # Restore cost_so_far and came_from if they exist
            if 'cost_so_far' in state:
                self.cost_so_far = state['cost_so_far']
            if 'came_from' in state:
                self.came_from = state['came_from']
    
    def start_planning(self):
        """Begin the path planning process using the selected algorithm."""
        if not self.start_cell or not self.goal_cell:
            print("Start or goal not set")
            return False
            
        self.reset()
        self.is_planning = True
        
        # Save the initial state to history if in step-by-step mode
        if self.step_by_step_mode:
            self._save_state_to_history()
            
        return True
        
    def heuristic(self, cell):
        """Calculate the heuristic value (estimated distance to goal)."""
        if not self.goal_cell:
            return 0
            
        # Using Euclidean distance for diagonal movement
        row, col = cell
        goal_row, goal_col = self.goal_cell
        
        # Euclidean distance
        dx = abs(col - goal_col)
        dy = abs(row - goal_row)
        return ((dx**2 + dy**2)**0.5)  # Euclidean distance
        
        # Alternative: Manhattan distance (better for orthogonal movement only)
        # return abs(row - goal_row) + abs(col - goal_col)
    
    def _execute_algorithm_step(self):
        """Execute one step of the selected algorithm."""
        if self.current_algorithm == Algorithm.DIJKSTRA:
            self._dijkstra_step()
        elif self.current_algorithm == Algorithm.ASTAR:
            self._astar_step()
        elif self.current_algorithm == Algorithm.BFS:
            self._bfs_step()
        elif self.current_algorithm == Algorithm.DFS:
            self._dfs_step()
        elif self.current_algorithm == Algorithm.RRT:
            self._rrt_step()
        elif self.current_algorithm == Algorithm.POTENTIAL_FIELD:
            self._potential_field_step()
            
        # Save the current state to history if in step-by-step mode
        if self.step_by_step_mode:
            self._save_state_to_history()
    
    def plan_step(self):
        """Execute a planning step based on the current mode."""
        if not self.is_planning or self.found_path:
            return
            
        # In automatic mode, execute the algorithm for several steps per frame
        if not self.step_by_step_mode:
            for _ in range(self.step_delay):
                self._execute_algorithm_step()
                if self.found_path or not self.is_planning:
                    break
                    
        # In step-by-step mode, we only execute steps when next_step() is called
    
    def _init_search(self):
        """Initialize the search data structures."""
        if self.current_algorithm == Algorithm.DIJKSTRA:
            # Priority queue: (cost, (row, col))
            self.pq = [(0, self.start_cell)]
            # Dictionary to store cost to reach each node
            self.cost_so_far = {self.start_cell: 0}
            
        elif self.current_algorithm == Algorithm.ASTAR:
            # Priority queue: (f_score, unique_id, (row, col))
            start_heuristic = self.heuristic(self.start_cell)
            self.counter = 0  # For unique identifiers
            self.pq = [(start_heuristic, self.counter, self.start_cell)]
            # Dictionary to store cost to reach each node
            self.cost_so_far = {self.start_cell: 0}
            
        elif self.current_algorithm == Algorithm.BFS:
            # Queue for BFS: just a list used as a queue
            self.queue = [self.start_cell]
            
        elif self.current_algorithm == Algorithm.DFS:
            # Stack for DFS: just a list used as a stack
            self.stack = [self.start_cell]

        elif self.current_algorithm == Algorithm.RRT:
            # Initialize RRT with start node
            self.rrt_nodes = [self.start_cell]
            self.rrt_edges = []
            self.rrt_iterations = 0
            
        elif self.current_algorithm == Algorithm.POTENTIAL_FIELD:
            # Initialize potential field
            self._init_potential_field()
            self.potential_path = [self.start_cell]
            self.potential_iterations = 0 

        # Common to all algorithms
        # Dictionary to store the path
        self.came_from = {self.start_cell: None}
        # Add start to frontier for visualization
        self.frontier.add(self.start_cell)
    
    def _dijkstra_step(self):
        """Execute one step of Dijkstra's algorithm."""
        # Initialize if this is the first step
        if not self.frontier:
            self._init_search()
            
        # Step through the algorithm
        if not self.pq:
            self.is_planning = False
            print("No path found!")
            return
            
        # Get the cell with the lowest cost
        current_cost, current_cell = heapq.heappop(self.pq)
        
        # Remove from frontier (for visualization)
        if current_cell in self.frontier:
            self.frontier.remove(current_cell)
            
        # Skip if already visited (might happen with priority queue updates)
        if current_cell in self.visited_cells:
            return
            
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
        self._explore_neighbors(current_cell, is_dijkstra=True)
    
    def _astar_step(self):
        """Execute one step of A* algorithm."""
        # Initialize if this is the first step
        if not self.frontier:
            self._init_search()
            
        # Step through the algorithm
        if not self.pq:
            self.is_planning = False
            print("No path found!")
            return
            
        # Get the cell with the lowest f_score
        f_score, _, current_cell = heapq.heappop(self.pq)
        
        # Remove from frontier (for visualization)
        if current_cell in self.frontier:
            self.frontier.remove(current_cell)
            
        # Skip if already visited (might happen with priority queue updates)
        if current_cell in self.visited_cells:
            return
            
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
        self._explore_neighbors(current_cell, is_astar=True)
    
    def _bfs_step(self):
        """Execute one step of Breadth-First Search algorithm."""
        # Initialize if this is the first step
        if not self.frontier:
            self._init_search()
            
        # Step through the algorithm
        if not self.queue:
            self.is_planning = False
            print("No path found!")
            return
            
        # Get the next cell from the queue (FIFO)
        current_cell = self.queue.pop(0)
        
        # Remove from frontier (for visualization)
        if current_cell in self.frontier:
            self.frontier.remove(current_cell)
            
        # Skip if already visited
        if current_cell in self.visited_cells:
            return
            
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
        self._explore_neighbors(current_cell, is_bfs=True)
    
    def _dfs_step(self):
        """Execute one step of Depth-First Search algorithm."""
        # Initialize if this is the first step
        if not self.frontier:
            self._init_search()
            
        # Step through the algorithm
        if not self.stack:
            self.is_planning = False
            print("No path found!")
            return
            
        # Get the next cell from the stack (LIFO)
        current_cell = self.stack.pop()
        
        # Remove from frontier (for visualization)
        if current_cell in self.frontier:
            self.frontier.remove(current_cell)
            
        # Skip if already visited
        if current_cell in self.visited_cells:
            return
            
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
        self._explore_neighbors(current_cell, is_dfs=True)
    
    def _explore_neighbors(self, current_cell, is_dijkstra=False, is_astar=False, is_bfs=False, is_dfs=False):
        """Explore neighbors based on the current algorithm."""
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
                not self.grid.is_cell_occupied(*next_cell) and
                next_cell not in self.visited_cells):
                
                # For Dijkstra's and A*: Process with costs
                if is_dijkstra or is_astar:
                    # Calculate new cost (1 for orthogonal, 1.414 for diagonal)
                    is_diagonal = (next_cell[0] != row and next_cell[1] != col)
                    movement_cost = 1.414 if is_diagonal else 1.0
                    new_cost = self.cost_so_far[current_cell] + movement_cost
                    
                    # If we haven't visited this cell or found a cheaper path
                    if next_cell not in self.cost_so_far or new_cost < self.cost_so_far[next_cell]:
                        self.cost_so_far[next_cell] = new_cost
                        
                        if is_astar:
                            # Calculate f_score = g_score + heuristic
                            f_score = new_cost + self.heuristic(next_cell)
                            
                            # Add to priority queue with f_score as priority
                            self.counter += 1  # Ensure unique ordering
                            heapq.heappush(self.pq, (f_score, self.counter, next_cell))
                        else:  # Dijkstra
                            heapq.heappush(self.pq, (new_cost, next_cell))
                        
                        self.came_from[next_cell] = current_cell
                        
                        # Add to frontier for visualization
                        self.frontier.add(next_cell)
                
                # For BFS: Add to queue if not visited
                elif is_bfs and next_cell not in self.queue and next_cell not in self.frontier:
                    self.queue.append(next_cell)
                    self.came_from[next_cell] = current_cell
                    self.frontier.add(next_cell)
                
                # For DFS: Add to stack if not visited
                elif is_dfs and next_cell not in self.stack and next_cell not in self.frontier:
                    self.stack.append(next_cell)
                    self.came_from[next_cell] = current_cell
                    self.frontier.add(next_cell)
    
    # RRT Implementation
    def _rrt_step(self):
        """Execute one step of the RRT algorithm."""
        # Initialize if this is the first step
        if not self.frontier:
            self._init_search()
            
        # Check if we've reached the maximum iterations
        if self.rrt_iterations >= self.rrt_max_iterations:
            self.is_planning = False
            print("RRT: Maximum iterations reached without finding a path.")
            return
            
        # Check if we've already found the path
        if self.found_path:
            return
            
        # Sample a random point
        if random.random() < self.rrt_goal_sample_rate:
            # Sample the goal with some probability
            sample = self.goal_cell
        else:
            # Otherwise sample a random point in the grid
            row = random.randint(0, self.grid.rows - 1)
            col = random.randint(0, self.grid.cols - 1)
            sample = (row, col)
            
        # Find the nearest node in the tree
        nearest_node = self._find_nearest_node(sample)
        
        # Extend the tree towards the sample
        new_node = self._extend_rrt(nearest_node, sample)
        
        # If extension was successful
        if new_node:
            # Add to visited for visualization
            self.visited_cells.add(new_node)
            
            # Add the edge to the tree
            self.rrt_edges.append((nearest_node, new_node))
            
            # Add the node to the tree
            self.rrt_nodes.append(new_node)
            
            # Save the parent information for path reconstruction
            self.came_from[new_node] = nearest_node
            
            # Check if we're close enough to the goal
            if self._distance(new_node, self.goal_cell) <= self.rrt_step_size:
                # Add the goal to the tree
                self.came_from[self.goal_cell] = new_node
                self.rrt_edges.append((new_node, self.goal_cell))
                self.rrt_nodes.append(self.goal_cell)
                
                # Reconstruct the path
                self.reconstruct_path()
                self.found_path = True
                self.is_planning = False
                print("RRT: Path found!")
                
        # Increment the iteration counter
        self.rrt_iterations += 1

    def _find_nearest_node(self, sample):
        """Find the node in the RRT that is closest to the sample."""
        nearest_node = None
        min_dist = float('inf')
        
        for node in self.rrt_nodes:
            dist = self._distance(node, sample)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
                
        return nearest_node

    def _extend_rrt(self, from_node, to_node):
        """Extend the RRT from the nearest node towards the sample."""
        # Calculate the direction vector
        row_diff = to_node[0] - from_node[0]
        col_diff = to_node[1] - from_node[1]
        
        # Calculate the distance
        dist = self._distance(from_node, to_node)
        
        # If the distance is less than the step size, just use the target
        if dist <= self.rrt_step_size:
            step_size = dist
        else:
            step_size = self.rrt_step_size
            
        # Normalize the direction vector and scale by step size
        if dist > 0:  # Avoid division by zero
            row_diff = row_diff * step_size / dist
            col_diff = col_diff * step_size / dist
            
        # Calculate the new node
        new_row = int(from_node[0] + row_diff)
        new_col = int(from_node[1] + col_diff)
        new_node = (new_row, new_col)
        
        # Check if the new node is valid (within bounds and not occupied)
        if (0 <= new_row < self.grid.rows and 
            0 <= new_col < self.grid.cols and 
            not self.grid.is_cell_occupied(new_row, new_col)):
            
            # Check if the path from the nearest node to the new node is collision-free
            if self._check_path(from_node, new_node):
                return new_node
                
        return None

    def _check_path(self, node1, node2):
        """Check if the path between two nodes is collision-free."""
        # Simple implementation: check a few points along the line
        points_to_check = 10
        
        for i in range(1, points_to_check):
            # Get a point along the line
            t = i / points_to_check
            row = int(node1[0] + t * (node2[0] - node1[0]))
            col = int(node1[1] + t * (node2[1] - node1[1]))
            
            # Check if this point is valid
            if self.grid.is_cell_occupied(row, col):
                return False
                
        return True

    def _distance(self, node1, node2):
        """Calculate the Euclidean distance between two nodes."""
        return ((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)**0.5

    # Potential Field Implementation
    def _init_potential_field(self):
        """Initialize the potential field grid."""
        # Create a grid to store potential values
        self.potential_field = [[0.0 for _ in range(self.grid.cols)] for _ in range(self.grid.rows)]
        
        # Set the goal position for the attractive field
        goal_row, goal_col = self.goal_cell
        
        # Calculate potential field for each cell
        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                # Skip occupied cells
                if self.grid.is_cell_occupied(row, col):
                    self.potential_field[row][col] = float('inf')
                    continue
                    
                # Calculate attractive potential (distance to goal)
                d_goal = ((row - goal_row)**2 + (col - goal_col)**2)**0.5
                attractive = self.attraction_factor * d_goal
                
                # Calculate repulsive potential (from obstacles)
                repulsive = 0.0
                
                # Check all cells within the repulsion radius
                for r in range(max(0, row - self.repulsion_radius), min(self.grid.rows, row + self.repulsion_radius + 1)):
                    for c in range(max(0, col - self.repulsion_radius), min(self.grid.cols, col + self.repulsion_radius + 1)):
                        if self.grid.is_cell_occupied(r, c):
                            d_obs = ((row - r)**2 + (col - c)**2)**0.5
                            if d_obs < self.repulsion_radius:
                                # Repulsive force increases as we get closer to obstacles
                                repulsive += self.repulsion_factor * (1.0 / d_obs - 1.0 / self.repulsion_radius)**2
                
                # Total potential is the sum of attractive and repulsive
                self.potential_field[row][col] = attractive + repulsive

    def _potential_field_step(self):
        """Execute one step of the potential field algorithm."""
        # Initialize if this is the first step
        if not self.frontier:
            self._init_search()
            
        # Check if we've reached the maximum iterations
        if self.potential_iterations >= self.potential_max_iterations:
            self.is_planning = False
            print("Potential Field: Maximum iterations reached.")
            return
            
        # Check if we've already found the path
        if self.found_path:
            return
            
        # Get the current position
        current = self.potential_path[-1]
        current_row, current_col = current
        
        # Add to visited for visualization
        self.visited_cells.add(current)
        
        # Check if we're close enough to the goal
        if self._distance(current, self.goal_cell) <= 1.5:
            # Add the goal to the path
            self.potential_path.append(self.goal_cell)
            
            # Set the path for visualization
            self.path = self.potential_path.copy()
            self.found_path = True
            self.is_planning = False
            print("Potential Field: Path found!")
            return
            
        # Find the cell with the lowest potential among neighbors
        min_potential = float('inf')
        next_cell = None
        
        # Check all 8 neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue  # Skip current cell
                    
                new_row = current_row + dr
                new_col = current_col + dc
                
                # Check if the cell is valid
                if (0 <= new_row < self.grid.rows and 
                    0 <= new_col < self.grid.cols and 
                    not self.grid.is_cell_occupied(new_row, new_col)):
                    
                    # Get the potential value
                    potential = self.potential_field[new_row][new_col]
                    
                    # If this is the lowest potential so far
                    if potential < min_potential:
                        min_potential = potential
                        next_cell = (new_row, new_col)
        
        # If we found a valid next cell
        if next_cell:
            # Add the next cell to the path
            self.potential_path.append(next_cell)
            
            # Update the path for visualization
            self.path = self.potential_path.copy()
            
            # Add to the frontier for visualization
            self.frontier.add(next_cell)
            
            # Save the parent information for visualization
            self.came_from[next_cell] = current
        else:
            # No valid next cell found, we're stuck
            self.is_planning = False
            print("Potential Field: Stuck in local minimum.")
            
        # Increment the iteration counter
        self.potential_iterations += 1

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
        
        # For Potential Field, visualize the field
        if self.current_algorithm == Algorithm.POTENTIAL_FIELD and self.potential_field:
            # Find the range of potential values for visualization
            min_potential = float('inf')
            max_potential = float('-inf')
            
            for row in range(self.grid.rows):
                for col in range(self.grid.cols):
                    pot = self.potential_field[row][col]
                    if pot != float('inf'):
                        min_potential = min(min_potential, pot)
                        max_potential = max(max_potential, pot)
            
            # Draw potential field as a heat map
            for row in range(self.grid.rows):
                for col in range(self.grid.cols):
                    pot = self.potential_field[row][col]
                    if pot != float('inf'):
                        # Normalize potential value
                        norm_pot = (pot - min_potential) / (max_potential - min_potential)
                        
                        # Create color gradient (blue -> purple -> red)
                        r = int(255 * norm_pot)
                        g = 0
                        b = int(255 * (1 - norm_pot))
                        
                        # Draw the cell with color
                        rect = pygame.Rect(
                            col * cell_size,
                            row * cell_size,
                            cell_size,
                            cell_size
                        )
                        s = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                        s.fill((r, g, b, 100))  # Semi-transparent
                        screen.blit(s, rect)
        
        # For RRT, draw the tree
        elif self.current_algorithm == Algorithm.RRT:
            # Draw the edges
            for from_node, to_node in self.rrt_edges:
                from_row, from_col = from_node
                to_row, to_col = to_node
                
                # Calculate positions
                from_pos = (from_col * cell_size + cell_size // 2, from_row * cell_size + cell_size // 2)
                to_pos = (to_col * cell_size + cell_size // 2, to_row * cell_size + cell_size // 2)
                
                # Draw the edge
                pygame.draw.line(screen, self.colors['rrt_edge'], from_pos, to_pos, 2)
            
            # Draw the nodes
            for node in self.rrt_nodes:
                row, col = node
                
                # Skip start and goal nodes (they'll be drawn separately)
                if node == self.start_cell or node == self.goal_cell:
                    continue
                    
                # Calculate position
                pos = (col * cell_size + cell_size // 2, row * cell_size + cell_size // 2)
                
                # Draw the node
                pygame.draw.circle(screen, self.colors['rrt_node'], pos, cell_size // 5)
        
        # Draw visited cells (for all algorithms)
        else:
            # Draw visited cells with cost and direction indicators
            if hasattr(self, 'came_from') and (hasattr(self, 'cost_so_far') or self.current_algorithm in [Algorithm.BFS, Algorithm.DFS]):
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
                    
                    # Draw cost value (for Dijkstra and A*)
                    if hasattr(self, 'cost_so_far') and cell in self.cost_so_far and self.show_decision_process:
                        cost = self.cost_so_far[cell]
                        if self.current_algorithm == Algorithm.ASTAR:
                            # For A*, show cost + heuristic
                            h_val = self.heuristic(cell)
                            cost_text = f"{cost:.1f}+{h_val:.1f}"
                        else:
                            # For Dijkstra, just show cost
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
            
            # Draw cost for frontier cells (for Dijkstra and A*)
            if hasattr(self, 'cost_so_far') and (row, col) in self.cost_so_far and self.show_decision_process:
                cost = self.cost_so_far[(row, col)]
                if self.current_algorithm == Algorithm.ASTAR:
                    # For A*, show cost + heuristic
                    h_val = self.heuristic((row, col))
                    cost_text = f"{cost:.1f}+{h_val:.1f}"
                else:
                    # For Dijkstra, just show cost
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
        
        # For algorithms that track cost
        if hasattr(self, 'cost_so_far') and self.goal_cell in self.cost_so_far:
            path_cost = self.cost_so_far[self.goal_cell]
        else:
            # For algorithms that don't track cost (BFS, DFS)
            path_cost = path_length
        
        return {
            'found': True,
            'length': path_length,
            'cost': path_cost,
            'visited_count': len(self.visited_cells)
        }
        
    def get_algorithm_name(self):
        """Return the name of the current algorithm."""
        return self.current_algorithm.name