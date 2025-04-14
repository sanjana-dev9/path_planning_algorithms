import pygame
import sys
from enum import Enum, auto
import pygame_gui

# from .algorithms.dijikstra import PathFinder  
from .algorithms.all import Algorithm, PathFinder

class Shape(Enum):
    NONE = auto()
    RECTANGLE = auto()
    CIRCLE = auto()
    TRIANGLE = auto()


class Button:
    def __init__(self, x, y, width, height, color, hover_color, text, text_color, action):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = hover_color
        self.text = text
        self.text_color = text_color
        self.action = action
        self.hovered = False

    def draw(self, screen, font):
        # Determine color based on hover state
        current_color = self.hover_color if self.hovered else self.color
        
        # Draw button rectangle
        pygame.draw.rect(screen, current_color, self.rect)
        pygame.draw.rect(screen, (200, 200, 200), self.rect, 2)  # Border
        
        # Draw text
        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def update(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered:
                self.action()
                return True
        return False


class Grid:
    def __init__(self, width, height, cell_size):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.cols = width // cell_size
        self.rows = height // cell_size
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.debug_mode = False

    def resize(self, width, height):
        """Resize the grid to match new window dimensions."""
        self.width = width
        self.height = height
        self.cols = width // self.cell_size
        self.rows = height // self.cell_size
        
        # Create new grid with updated dimensions
        new_grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Copy existing grid data where possible
        for r in range(min(len(self.grid), self.rows)):
            for c in range(min(len(self.grid[0]), self.cols)):
                new_grid[r][c] = self.grid[r][c]
        
        self.grid = new_grid

    def update_from_obstacles(self, obstacles):
        """Update grid occupancy based on obstacle positions."""
        # Reset grid
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Mark cells as occupied based on obstacles
        for obstacle in obstacles:
            self.mark_obstacle(obstacle)
    
    def mark_obstacle(self, obstacle):
        """Mark grid cells covered by an obstacle as occupied."""
        if obstacle.shape_type == Shape.RECTANGLE:
            x, y, w, h = obstacle.points
            # Include buffer zone
            x_with_buffer = max(0, x - obstacle.buffer_zone)
            y_with_buffer = max(0, y - obstacle.buffer_zone)
            w_with_buffer = w + 2 * obstacle.buffer_zone
            h_with_buffer = h + 2 * obstacle.buffer_zone
            
            # Convert to grid cells
            start_col = max(0, int(x_with_buffer // self.cell_size))
            start_row = max(0, int(y_with_buffer // self.cell_size))
            end_col = min(self.cols - 1, int((x_with_buffer + w_with_buffer) // self.cell_size) + 1)
            end_row = min(self.rows - 1, int((y_with_buffer + h_with_buffer) // self.cell_size) + 1)
            
            # Mark cells
            for row in range(start_row, end_row + 1):
                for col in range(start_col, end_col + 1):
                    if 0 <= row < self.rows and 0 <= col < self.cols:
                        self.grid[row][col] = 1
        
        elif obstacle.shape_type == Shape.CIRCLE:
            cx, cy, radius = obstacle.points
            radius_with_buffer = radius + obstacle.buffer_zone
            
            # Determine affected grid area
            start_col = max(0, int((cx - radius_with_buffer) // self.cell_size))
            start_row = max(0, int((cy - radius_with_buffer) // self.cell_size))
            end_col = min(self.cols - 1, int((cx + radius_with_buffer) // self.cell_size) + 1)
            end_row = min(self.rows - 1, int((cy + radius_with_buffer) // self.cell_size) + 1)
            
            # Check each cell if it intersects with the circle
            for row in range(start_row, end_row + 1):
                for col in range(start_col, end_col + 1):
                    # Test the four corners and center of the cell
                    cell_x = col * self.cell_size
                    cell_y = row * self.cell_size
                    
                    # Test points: center and four corners of the cell
                    test_points = [
                        (cell_x + self.cell_size/2, cell_y + self.cell_size/2),  # center
                        (cell_x, cell_y),  # top-left
                        (cell_x + self.cell_size, cell_y),  # top-right
                        (cell_x, cell_y + self.cell_size),  # bottom-left
                        (cell_x + self.cell_size, cell_y + self.cell_size)  # bottom-right
                    ]
                    
                    # If any point is inside the circle, mark cell as occupied
                    for px, py in test_points:
                        dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
                        if dist <= radius_with_buffer:
                            if 0 <= row < self.rows and 0 <= col < self.cols:
                                self.grid[row][col] = 1
                            break
        
        elif obstacle.shape_type == Shape.TRIANGLE:
            # For triangle, we'll use a simplified approach - check cells against triangle edges
            p1, p2, p3 = obstacle.points
            
            # Find bounding box of the triangle with buffer
            min_x = min(p1[0], p2[0], p3[0]) - obstacle.buffer_zone
            min_y = min(p1[1], p2[1], p3[1]) - obstacle.buffer_zone
            max_x = max(p1[0], p2[0], p3[0]) + obstacle.buffer_zone
            max_y = max(p1[1], p2[1], p3[1]) + obstacle.buffer_zone
            
            # Convert to grid cells
            start_col = max(0, int(min_x // self.cell_size))
            start_row = max(0, int(min_y // self.cell_size))
            end_col = min(self.cols - 1, int(max_x // self.cell_size) + 1)
            end_row = min(self.rows - 1, int(max_y // self.cell_size) + 1)
            
            # Check each cell if it intersects with the triangle
            for row in range(start_row, end_row + 1):
                for col in range(start_col, end_col + 1):
                    # Get cell center
                    cell_x = col * self.cell_size + self.cell_size/2
                    cell_y = row * self.cell_size + self.cell_size/2
                    
                    # Use our obstacle collision detection
                    if obstacle.collides_with_point((cell_x, cell_y), include_buffer=True):
                        if 0 <= row < self.rows and 0 <= col < self.cols:
                            self.grid[row][col] = 1

    def is_cell_occupied(self, row, col):
        """Check if a specific cell is occupied."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col] == 1
        return True  # Treat out-of-bounds as occupied

    def get_cell(self, x, y):
        """Get grid cell at pixel coordinates."""
        col = int(x // self.cell_size)
        row = int(y // self.cell_size)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return (row, col)
        return None
    
    def is_occupied(self, x, y):
        """Check if the cell at pixel coordinates is occupied."""
        cell = self.get_cell(x, y)
        if cell:
            row, col = cell
            return self.grid[row][col] == 1
        return True  # Treat out-of-bounds as occupied
    
    def draw(self, screen):
        """Draw the grid on the screen."""
        # Draw grid lines
        for i in range(self.rows + 1):
            y = i * self.cell_size
            pygame.draw.line(screen, (60, 60, 60), (0, y), (self.width, y), 1)
        
        for j in range(self.cols + 1):
            x = j * self.cell_size
            pygame.draw.line(screen, (60, 60, 60), (x, 0), (x, self.height), 1)
        
        # In debug mode, visualize occupied cells
        if self.debug_mode:
            for row in range(self.rows):
                for col in range(self.cols):
                    if self.grid[row][col] == 1:
                        # Draw a semi-transparent rectangle over occupied cells
                        cell_rect = pygame.Rect(
                            col * self.cell_size, 
                            row * self.cell_size,
                            self.cell_size, 
                            self.cell_size
                        )
                        # Red with transparency
                        s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                        s.fill((255, 0, 0, 50))  # Transparent red
                        screen.blit(s, cell_rect)


class Obstacle:
    def __init__(self, shape_type, points, color=(255, 0, 0), buffer_zone=20):
        self.shape_type = shape_type
        self.points = points
        self.color = color
        self.buffer_zone = buffer_zone  # Size of no breach zone in pixels
        self.buffer_color = (150, 150, 0, 80)  # Yellow-ish with transparency

    def draw(self, screen):
        # Draw the buffer zone first (so it appears behind the obstacle)
        self.draw_buffer_zone(screen)
        
        # Draw the main obstacle
        if self.shape_type == Shape.RECTANGLE:
            pygame.draw.rect(screen, self.color, pygame.Rect(*self.points))
        elif self.shape_type == Shape.CIRCLE:
            pygame.draw.circle(screen, self.color, self.points[0:2], self.points[2])
        elif self.shape_type == Shape.TRIANGLE:
            pygame.draw.polygon(screen, self.color, self.points)

    def draw_buffer_zone(self, screen):
        # Create a transparent surface for the buffer zone
        buffer_surface = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
        
        if self.shape_type == Shape.RECTANGLE:
            x, y, w, h = self.points
            # Draw the expanded rectangle for buffer zone
            buffer_rect = pygame.Rect(
                x - self.buffer_zone, 
                y - self.buffer_zone,
                w + 2 * self.buffer_zone,
                h + 2 * self.buffer_zone
            )
            pygame.draw.rect(buffer_surface, self.buffer_color, buffer_rect)
            
        elif self.shape_type == Shape.CIRCLE:
            cx, cy, radius = self.points
            # Draw larger circle for buffer zone
            pygame.draw.circle(buffer_surface, self.buffer_color, (cx, cy), radius + self.buffer_zone)
            
        elif self.shape_type == Shape.TRIANGLE:
            # For triangle, we create a buffer by drawing lines with thickness
            # and then filling the area with a polygon that encompasses the original
            p1, p2, p3 = self.points
            
            # Calculate normals (perpendicular vectors) for each edge
            def normalize(v):
                length = (v[0]**2 + v[1]**2)**0.5
                return (v[0]/length, v[1]/length) if length > 0 else (0, 0)
                
            def get_normal(p1, p2):
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                # Perpendicular vector
                nx, ny = -dy, dx
                # Normalize and scale by buffer zone
                normal = normalize((nx, ny))
                return normal[0] * self.buffer_zone, normal[1] * self.buffer_zone
            
            # Get normals for each edge
            n12 = get_normal(p1, p2)
            n23 = get_normal(p2, p3)
            n31 = get_normal(p3, p1)
            
            # Calculate expanded points
            buffer_points = [
                (p1[0] + n12[0] + n31[0], p1[1] + n12[1] + n31[1]),
                (p2[0] + n12[0] + n23[0], p2[1] + n12[1] + n23[1]),
                (p3[0] + n23[0] + n31[0], p3[1] + n23[1] + n31[1])
            ]
            
            pygame.draw.polygon(buffer_surface, self.buffer_color, buffer_points)
        
        # Blit the buffer surface onto the main screen
        screen.blit(buffer_surface, (0, 0))

    def collides_with_point(self, point, include_buffer=True):
        x, y = point
        
        # First check if point is in the main obstacle
        main_collision = self._point_in_main_obstacle(point)
        if main_collision:
            return True
            
        # If we're including the buffer zone in collision detection
        if include_buffer:
            return self._point_in_buffer_zone(point)
        
        return False
        
    def _point_in_main_obstacle(self, point):
        x, y = point
        if self.shape_type == Shape.RECTANGLE:
            rect = pygame.Rect(*self.points)
            return rect.collidepoint(x, y)
        elif self.shape_type == Shape.CIRCLE:
            cx, cy, radius = self.points
            distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            return distance <= radius
        elif self.shape_type == Shape.TRIANGLE:
            # Using barycentric coordinates to check if point is inside triangle
            p1, p2, p3 = self.points
            def area(p1, p2, p3):
                return abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))/2.0)
            
            A = area(p1, p2, p3)
            A1 = area(point, p2, p3)
            A2 = area(p1, point, p3)
            A3 = area(p1, p2, point)
            
            return abs(A - (A1 + A2 + A3)) < 0.1  # Small epsilon for float comparison
            
    def _point_in_buffer_zone(self, point):
        x, y = point
        if self.shape_type == Shape.RECTANGLE:
            x_rect, y_rect, w, h = self.points
            buffer_rect = pygame.Rect(
                x_rect - self.buffer_zone,
                y_rect - self.buffer_zone,
                w + 2 * self.buffer_zone,
                h + 2 * self.buffer_zone
            )
            return buffer_rect.collidepoint(x, y)
            
        elif self.shape_type == Shape.CIRCLE:
            cx, cy, radius = self.points
            distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            return distance <= (radius + self.buffer_zone)
            
        elif self.shape_type == Shape.TRIANGLE:
            # For the triangle buffer, we perform distance checks from edges
            p1, p2, p3 = self.points
            
            # Function to calculate distance from point to line segment
            def dist_point_to_segment(p, a, b):
                # Vector from a to b
                ab = (b[0] - a[0], b[1] - a[1])
                # Vector from a to p
                ap = (p[0] - a[0], p[1] - a[1])
                
                # Length of ab squared
                ab_squared = ab[0]**2 + ab[1]**2
                
                # If ab is a zero-length line, distance is just distance from p to a
                if ab_squared == 0:
                    return ((p[0] - a[0])**2 + (p[1] - a[1])**2)**0.5
                
                # Calculate dot product
                dot_product = ap[0] * ab[0] + ap[1] * ab[1]
                
                # Calculate projection of ap onto ab
                t = max(0, min(1, dot_product / ab_squared))
                
                # Calculate closest point on the line segment
                closest_x = a[0] + t * ab[0]
                closest_y = a[1] + t * ab[1]
                
                # Return distance to closest point
                return ((p[0] - closest_x)**2 + (p[1] - closest_y)**2)**0.5
            
            # Check distance to each edge
            dist1 = dist_point_to_segment(point, p1, p2)
            dist2 = dist_point_to_segment(point, p2, p3)
            dist3 = dist_point_to_segment(point, p3, p1)
            
            # If any distance is within buffer zone, there's a collision
            return min(dist1, dist2, dist3) <= self.buffer_zone


def main():
    # Initialize Pygame
    pygame.init()
    
    # Initial window size
    INITIAL_WIDTH, INITIAL_HEIGHT = 1000, 700
    TOOLBAR_HEIGHT = 120  # Increased for two rows of buttons
    
    # Get screen info for adaptive sizing
    screen_info = pygame.display.Info()
    max_width, max_height = screen_info.current_w, screen_info.current_h
    
    # Set window size based on screen size (use 80% of screen size or initial size, whichever is smaller)
    WIDTH = min(int(max_width * 0.8), INITIAL_WIDTH)
    HEIGHT = min(int((max_height - TOOLBAR_HEIGHT) * 0.8), INITIAL_HEIGHT)
    
    # Grid cell size
    CELL_SIZE = 100  # Size of each grid cell in pixels
    
    # Other constants
    DOT_RADIUS = 15
    DOT_COLOR = (0, 255, 0)  # Green
    BG_COLOR = (30, 30, 30)  # Dark gray
    MOVE_SPEED = 5
    BUTTON_WIDTH = 90
    BUTTON_HEIGHT = 35
    BUTTON_COLOR = (70, 70, 70)
    BUTTON_HOVER_COLOR = (100, 100, 100)
    BUTTON_TEXT_COLOR = (255, 255, 255)
    OBSTACLE_COLOR = (255, 100, 100)
    BUFFER_ZONE_SIZE = 20  # Size of no breach zone in pixels
    
    # Make window resizable
    screen = pygame.display.set_mode((WIDTH, HEIGHT + TOOLBAR_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Path Planning Algorithms for Static Maps")

    # Create a UI Manager
    ui_manager = pygame_gui.UIManager((WIDTH, HEIGHT + TOOLBAR_HEIGHT), 'theme.json')
    
    # Initialize grid
    grid = Grid(WIDTH, HEIGHT, CELL_SIZE)
    
    # Initialize path planner
    path_finder = PathFinder(grid)
    
    # Font for buttons
    font = pygame.font.SysFont('Arial', 15)
    
    # Initial position of the dot (center of the window)
    dot_x = WIDTH // 2
    dot_y = HEIGHT // 2
    
    # Create buttons list
    buttons = []
    
    # Current selected shape
    current_shape = Shape.NONE
    
    # List to store obstacles
    obstacles = []
    
    # Path planning mode flags
    placing_start = False
    placing_goal = False
    
    # Drawing state
    drawing = False
    start_pos = None

    # Create algorithm dropdown options
    algorithm_options = ["A*", "Dijkstra", "BFS", "DFS", "RRT", "Potential Field"]
    algorithm_dropdown = pygame_gui.elements.UIDropDownMenu(
        options_list=algorithm_options,
        starting_option=algorithm_options[0],
        relative_rect=pygame.Rect((5, HEIGHT + (2 * BUTTON_HEIGHT) + 10), (140, 30)), 
        manager=ui_manager
    )
    
    # Function to set current shape
    def set_shape(shape):
        nonlocal current_shape, placing_start, placing_goal
        current_shape = shape
        placing_start = False
        placing_goal = False
    
    # Set path planning modes
    def set_place_start():
        nonlocal current_shape, placing_start, placing_goal
        current_shape = Shape.NONE
        placing_start = True
        placing_goal = False
    
    def set_place_goal():
        nonlocal current_shape, placing_start, placing_goal
        current_shape = Shape.NONE
        placing_start = False
        placing_goal = True

    # Function to set algorithm based on dropdown selection
    def update_algorithm(selected):
        if selected[0] == "A*":
            path_finder.set_algorithm(Algorithm.ASTAR)
        elif selected[0] == "Dijkstra":
            path_finder.set_algorithm(Algorithm.DIJKSTRA)
        elif selected[0] == "BFS":
            path_finder.set_algorithm(Algorithm.BFS)
        elif selected[0] == "DFS":
            path_finder.set_algorithm(Algorithm.DFS)
        elif selected[0] == "RRT":
            path_finder.set_algorithm(Algorithm.RRT)
        elif selected[0] == "Potential Field":
            path_finder.set_algorithm(Algorithm.POTENTIAL_FIELD)
        
    def start_planning():
        nonlocal current_shape, placing_start, placing_goal
        current_shape = Shape.NONE
        placing_start = False
        placing_goal = False
        path_finder.start_planning()
    
    def reset_planning():
        nonlocal current_shape, placing_start, placing_goal
        current_shape = Shape.NONE
        placing_start = False
        placing_goal = False
        path_finder.reset()
    
    # Toggle for visualization options
    show_buffer = True
    
    def toggle_buffer_visibility():
        nonlocal show_buffer
        show_buffer = not show_buffer
    
    # Toggle grid debug mode
    def toggle_grid_debug():
        grid.debug_mode = not grid.debug_mode
        
    # Toggle decision process visualization
    def toggle_decision_process():
        path_finder.show_decision_process = not path_finder.show_decision_process
        
    # Toggle step-by-step mode
    def toggle_step_by_step():
        path_finder.toggle_step_by_step_mode()
        
    # Move to next step in step-by-step mode
    def next_step():
        path_finder.next_step()
        
    # Move to previous step in step-by-step mode
    def prev_step():
        path_finder.prev_step()
    
    # Function to update button positions when window is resized
    def update_buttons():
        buttons.clear()
        
        button_spacing = 5
        button_x = button_spacing
        button_row1_y = HEIGHT + 5
        button_row2_y = HEIGHT + BUTTON_HEIGHT + 10

        # Row 1: Obstacle creation tools
        # Rectangle button
        rect_button = Button(
            button_x, button_row1_y, BUTTON_WIDTH, BUTTON_HEIGHT, 
            BUTTON_COLOR, BUTTON_HOVER_COLOR, "Rectangle", BUTTON_TEXT_COLOR,
            lambda: set_shape(Shape.RECTANGLE)
        )
        buttons.append(rect_button)
        button_x += BUTTON_WIDTH + button_spacing
        
        # Circle button
        circle_button = Button(
            button_x, button_row1_y, BUTTON_WIDTH, BUTTON_HEIGHT, 
            BUTTON_COLOR, BUTTON_HOVER_COLOR, "Circle", BUTTON_TEXT_COLOR,
            lambda: set_shape(Shape.CIRCLE)
        )
        buttons.append(circle_button)
        button_x += BUTTON_WIDTH + button_spacing
        
        # Triangle button
        triangle_button = Button(
            button_x, button_row1_y, BUTTON_WIDTH, BUTTON_HEIGHT, 
            BUTTON_COLOR, BUTTON_HOVER_COLOR, "Triangle", BUTTON_TEXT_COLOR,
            lambda: set_shape(Shape.TRIANGLE)
        )
        buttons.append(triangle_button)
        button_x += BUTTON_WIDTH + button_spacing
        
        # Buffer toggle button
        buffer_toggle_button = Button(
            button_x, button_row1_y, BUTTON_WIDTH, BUTTON_HEIGHT,
            (70, 100, 70), (100, 150, 100), "Toggle Buffer", BUTTON_TEXT_COLOR,
            toggle_buffer_visibility
        )
        buttons.append(buffer_toggle_button)
        button_x += BUTTON_WIDTH + button_spacing
        
        # Grid debug button
        grid_debug_button = Button(
            button_x, button_row1_y, BUTTON_WIDTH, BUTTON_HEIGHT,
            (70, 70, 100), (100, 100, 150), "Grid Debug", BUTTON_TEXT_COLOR,
            toggle_grid_debug
        )
        buttons.append(grid_debug_button)
        button_x += BUTTON_WIDTH + button_spacing
        
        # Clear button
        clear_button = Button(
            button_x, button_row1_y, BUTTON_WIDTH, BUTTON_HEIGHT, 
            (150, 50, 50), (200, 70, 70), "Clear All", BUTTON_TEXT_COLOR,
            lambda: (obstacles.clear(), path_finder.reset())
        )
        buttons.append(clear_button)
        
        # Row 2: Path planning buttons
        button_x = button_spacing
        
        # Set Start button
        start_button = Button(
            button_x, button_row2_y, BUTTON_WIDTH, BUTTON_HEIGHT,
            (50, 150, 50), (70, 200, 70), "Set Start", BUTTON_TEXT_COLOR,
            set_place_start
        )
        buttons.append(start_button)
        button_x += BUTTON_WIDTH + button_spacing
        
        # Set Goal button
        goal_button = Button(
            button_x, button_row2_y, BUTTON_WIDTH, BUTTON_HEIGHT,
            (150, 50, 50), (200, 70, 70), "Set Goal", BUTTON_TEXT_COLOR,
            set_place_goal
        )
        buttons.append(goal_button)
        button_x += BUTTON_WIDTH + button_spacing
        
        # Reset Planning button
        reset_plan_button = Button(
            button_x, button_row2_y, BUTTON_WIDTH, BUTTON_HEIGHT,
            (150, 100, 50), (200, 130, 70), "Reset Path", BUTTON_TEXT_COLOR,
            reset_planning
        )
        buttons.append(reset_plan_button)
        button_x += BUTTON_WIDTH + button_spacing
        
        # Toggle Decision Process button
        decision_process_button = Button(
            button_x, button_row2_y, BUTTON_WIDTH, BUTTON_HEIGHT,
            (100, 100, 100), (150, 150, 150), "Toggle Costs", BUTTON_TEXT_COLOR,
            toggle_decision_process
        )
        buttons.append(decision_process_button)
        button_x += BUTTON_WIDTH + button_spacing
        
        # Step-by-step mode button
        step_button = Button(
            button_x, button_row2_y, BUTTON_WIDTH, BUTTON_HEIGHT,
            (100, 50, 150), (150, 100, 200), "Step Mode", BUTTON_TEXT_COLOR,
            toggle_step_by_step
        )
        buttons.append(step_button)
        
        # If there's room, add previous and next step buttons
        if WIDTH >= 900:
            button_x += BUTTON_WIDTH + button_spacing
            prev_button = Button(
                button_x, button_row2_y, BUTTON_WIDTH // 2, BUTTON_HEIGHT,
                (70, 70, 100), (100, 100, 150), "←", BUTTON_TEXT_COLOR,
                prev_step
            )
            buttons.append(prev_button)
            
            next_button = Button(
                button_x + BUTTON_WIDTH // 2, button_row2_y, BUTTON_WIDTH // 2, BUTTON_HEIGHT,
                (70, 70, 100), (100, 100, 150), "→", BUTTON_TEXT_COLOR,
                next_step
            )
            buttons.append(next_button)
        
        button_x += BUTTON_WIDTH + button_spacing
        
        # Start Planning button
        plan_button = Button(
            button_x, button_row2_y, BUTTON_WIDTH, BUTTON_HEIGHT,
            (50, 50, 150), (70, 70, 200), "Find Path", BUTTON_TEXT_COLOR,
            start_planning
        )
        buttons.append(plan_button)
        
    # Initialize buttons
    update_buttons()
    
    # Triangle points
    triangle_points = []
    
    # Game loop
    clock = pygame.time.Clock()
    running = True

    while running:
        time_delta = clock.tick(60)/1000.0
        mouse_pos = pygame.mouse.get_pos()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Handle window resize events
            elif event.type == pygame.VIDEORESIZE:
                # Update WIDTH and HEIGHT
                WIDTH, screen_height = event.size
                HEIGHT = screen_height - TOOLBAR_HEIGHT
                
                # Update screen with new size
                screen = pygame.display.set_mode((WIDTH, HEIGHT + TOOLBAR_HEIGHT), pygame.RESIZABLE)
                
                # Update UI manager with new dimensions
                ui_manager.set_window_resolution(event.size)
                
                # Update dropdown position to keep it at the bottom
                algorithm_dropdown.set_relative_position((5, HEIGHT + (2 * BUTTON_HEIGHT) + 10))

                # Resize grid
                grid.resize(WIDTH, HEIGHT)
                
                # Keep dot within new boundaries
                dot_x = min(max(DOT_RADIUS, dot_x), WIDTH - DOT_RADIUS)
                dot_y = min(max(DOT_RADIUS, dot_y), HEIGHT - DOT_RADIUS)
                
                # Update button positions
                update_buttons()

            
            # Process UI events
            ui_manager.process_events(event)

            # Handle dropdown selection
            if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                if event.ui_element == algorithm_dropdown:
                    selected_algorithm = algorithm_dropdown.selected_option
                    
                    # Update your algorithm based on selection
                    update_algorithm(selected_algorithm)

            # Handle button events
            for button in buttons:
                if button.handle_event(event):
                    break
            
            # Handle mouse events for placing start/goal
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if mouse_pos[1] < HEIGHT:  # Only in the grid area, not toolbar
                    if placing_start:
                        if path_finder.set_start(mouse_pos[0], mouse_pos[1]):
                            placing_start = False
                    elif placing_goal:
                        if path_finder.set_goal(mouse_pos[0], mouse_pos[1]):
                            placing_goal = False
            
            # Handle mouse events for drawing obstacles
            if current_shape != Shape.NONE:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Check if click is in drawing area (not in toolbar)
                    if mouse_pos[1] < HEIGHT:
                        if current_shape == Shape.TRIANGLE:
                            if len(triangle_points) < 3:
                                triangle_points.append(mouse_pos)
                                if len(triangle_points) == 3:
                                    obstacles.append(Obstacle(Shape.TRIANGLE, triangle_points, OBSTACLE_COLOR, BUFFER_ZONE_SIZE))
                                    triangle_points = []
                                    current_shape = Shape.NONE
                                    # Update grid
                                    grid.update_from_obstacles(obstacles)
                        else:
                            drawing = True
                            start_pos = mouse_pos
                
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if drawing and current_shape != Shape.TRIANGLE:
                        end_pos = mouse_pos
                        if current_shape == Shape.RECTANGLE:
                            # Calculate rectangle coordinates
                            x = min(start_pos[0], end_pos[0])
                            y = min(start_pos[1], end_pos[1])
                            width = abs(start_pos[0] - end_pos[0])
                            height = abs(start_pos[1] - end_pos[1])
                            if width > 5 and height > 5:  # Minimum size check
                                obstacles.append(Obstacle(Shape.RECTANGLE, (x, y, width, height), OBSTACLE_COLOR, BUFFER_ZONE_SIZE))
                                # Update grid
                                grid.update_from_obstacles(obstacles)
                        elif current_shape == Shape.CIRCLE:
                            # Calculate circle parameters
                            center_x = start_pos[0]
                            center_y = start_pos[1]
                            radius = ((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)**0.5
                            if radius > 5:  # Minimum size check
                                obstacles.append(Obstacle(Shape.CIRCLE, (center_x, center_y, radius), OBSTACLE_COLOR, BUFFER_ZONE_SIZE))
                                # Update grid
                                grid.update_from_obstacles(obstacles)
                        
                        drawing = False
                        current_shape = Shape.NONE
        
        # Update button hover states
        for button in buttons:
            button.update(mouse_pos)

        # Update dropdown hover state
        ui_manager.update(time_delta)
        
        # Get the state of all keyboard keys
        keys = pygame.key.get_pressed()
        
        # Store previous position
        prev_x, prev_y = dot_x, dot_y
        
        # Move the dot based on WASD keys
        if keys[pygame.K_w]:
            dot_y -= MOVE_SPEED
        if keys[pygame.K_s]:
            dot_y += MOVE_SPEED
        if keys[pygame.K_a]:
            dot_x -= MOVE_SPEED
        if keys[pygame.K_d]:
            dot_x += MOVE_SPEED
        
        # Keep the dot within the screen boundaries
        dot_x = max(DOT_RADIUS, min(dot_x, WIDTH - DOT_RADIUS))
        dot_y = max(DOT_RADIUS, min(dot_y, HEIGHT - DOT_RADIUS))
        
        # Check for collision with obstacles and grid
        # First check direct obstacle collision (for non-grid based objects)
        dot_collides = False
        for obstacle in obstacles:
            if obstacle.collides_with_point((dot_x, dot_y), include_buffer=True):
                dot_collides = True
                break
        
        # Additionally, check grid cell occupancy
        if not dot_collides and grid.is_occupied(dot_x, dot_y):
            dot_collides = True
        
        # If collision detected, revert to previous position
        if dot_collides:
            dot_x, dot_y = prev_x, prev_y
        
        # Execute path planning step if active
        if path_finder.is_planning:
            path_finder.plan_step()
        
        # Clear the screen
        screen.fill(BG_COLOR)
        
        # Draw the grid
        grid.draw(screen)
        
        # Draw the path finder visualization
        path_finder.draw(screen)
        
        # Draw the obstacles
        for obstacle in obstacles:
            # Only show buffer if the toggle is on
            if show_buffer:
                obstacle.draw(screen)
            else:
                # Just draw the main obstacle without buffer
                if obstacle.shape_type == Shape.RECTANGLE:
                    pygame.draw.rect(screen, obstacle.color, pygame.Rect(*obstacle.points))
                elif obstacle.shape_type == Shape.CIRCLE:
                    pygame.draw.circle(screen, obstacle.color, obstacle.points[0:2], obstacle.points[2])
                elif obstacle.shape_type == Shape.TRIANGLE:
                    pygame.draw.polygon(screen, obstacle.color, obstacle.points)
        
        # Draw preview while drawing
        if drawing:
            end_pos = mouse_pos
            if current_shape == Shape.RECTANGLE:
                x = min(start_pos[0], end_pos[0])
                y = min(start_pos[1], end_pos[1])
                width = abs(start_pos[0] - end_pos[0])
                height = abs(start_pos[1] - end_pos[1])
                pygame.draw.rect(screen, (*OBSTACLE_COLOR, 128), (x, y, width, height))
                pygame.draw.rect(screen, (255, 255, 255), (x, y, width, height), 1)
            elif current_shape == Shape.CIRCLE:
                center_x = start_pos[0]
                center_y = start_pos[1]
                radius = ((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)**0.5
                pygame.draw.circle(screen, (*OBSTACLE_COLOR, 128), (center_x, center_y), radius)
                pygame.draw.circle(screen, (255, 255, 255), (center_x, center_y), radius, 1)
        
        # Draw triangle points being placed
        for i, point in enumerate(triangle_points):
            pygame.draw.circle(screen, OBSTACLE_COLOR, point, 5)
            if i > 0:
                pygame.draw.line(screen, OBSTACLE_COLOR, triangle_points[i-1], point, 2)
        
        if len(triangle_points) == 2:
            pygame.draw.line(screen, (*OBSTACLE_COLOR, 128), triangle_points[-1], mouse_pos, 2)
        
        # Draw the dot (if we're not in path planning mode)
        if not path_finder.is_planning and not path_finder.found_path:
            pygame.draw.circle(screen, DOT_COLOR, (dot_x, dot_y), DOT_RADIUS)
        
        # Draw toolbar background
        pygame.draw.rect(screen, (50, 50, 50), (0, HEIGHT, WIDTH, TOOLBAR_HEIGHT))
        
        # Draw the buttons
        for button in buttons:
            button.draw(screen, font)
        
        # Mode info text
        info_x = WIDTH - 10
        info_y = HEIGHT + BUTTON_HEIGHT + 10
        
        # Draw current mode indicators
        mode_text = ""
        if placing_start:
            mode_text = "Click to place Start point"
        elif placing_goal:
            mode_text = "Click to place Goal point"
        elif current_shape != Shape.NONE:
            mode_text = f"Current Tool: {current_shape.name}"
        elif path_finder.is_planning:
            mode_text = "Planning in progress..."
        elif path_finder.found_path:
            mode_text = "Path found!"
        else:
            mode_text = "Select a tool or set Start/Goal"
            
        mode_surf = font.render(mode_text, True, (200, 200, 200))
        mode_rect = mode_surf.get_rect(right=info_x, top=info_y)
        screen.blit(mode_surf, mode_rect)

        # Draw algorithm info
        algo_text = f"Algorithm: {path_finder.get_algorithm_name()}"
        algo_surf = font.render(algo_text, True, (200, 200, 200))
        algo_rect = algo_surf.get_rect(right=info_x, top=info_y - 20)
        screen.blit(algo_surf, algo_rect)
        
        # Draw step-by-step mode info
        step_mode_text = f"Step Mode: {'ON' if path_finder.step_by_step_mode else 'OFF'}"
        step_surf = font.render(step_mode_text, True, (200, 200, 200))
        step_rect = step_surf.get_rect(right=info_x, top=info_y - 40)
        screen.blit(step_surf, step_rect)
        
        buffer_status = f"Buffer Zone: {'ON' if show_buffer else 'OFF'} (still active for collisions)"
        buffer_surf = font.render(buffer_status, True, (200, 200, 200))
        buffer_rect = buffer_surf.get_rect(right=info_x, top=info_y + 20)
        screen.blit(buffer_surf, buffer_rect)
        
        # Add window size info
        size_text = f"Window: {WIDTH}x{HEIGHT} | Grid: {grid.cols}x{grid.rows} | Cell Size: {CELL_SIZE}px"
        size_surf = font.render(size_text, True, (200, 200, 200))
        size_rect = size_surf.get_rect(right=info_x, top=info_y + 40)
        screen.blit(size_surf, size_rect)
        
        # Draw UI
        ui_manager.draw_ui(screen)

        # Update the display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(60)

    # Clean up
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()