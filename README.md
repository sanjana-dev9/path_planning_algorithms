# Path Planning Algorithms

Visualise how different algorithms navigate in a map

[Youtube] (https://www.youtube.com/watch?v=UtOh1nryxEI)

[Demo video.webm](https://github.com/user-attachments/assets/d39e2a8b-136d-4b94-9bf7-a65268ce8e73)

# Setup

-   python or C++
-   Initialise poetry: `poetry init`

-   To activate the virtual env created by poetry: `poetry shell`
-   Poetry to automatically update the packages

    > Install dependencies: `poetry add package-name` <br>
    > Install dev dependencies: `poetry add --group dev package-name` <br>
    > Install your project: `poetry install` <br>
    > Update dependencies: `poetry update` <br>

<br>
<br>

# Todos

<br>

## Basic

1. Create a 2d map with a moving dot
2. Dot can navigate in this space
3. Load a map with obstacles
4. Create dijkstra graph out of this map
5. Visualise the path if the dot is at one of the node and has to travel to other node
6. Step 2 will actiavte once path is generated
   <br>

## ROS2:

[Github link for turtlesim](https://github.com/sanjana-dev9/turtlesim)

[Github link for turtlebot](https://github.com/sanjana-dev9/turtlebot)

1. Integrate turtlesim and turtelboot
2. Support ROS2
3. Visualise ROS2 SLAM packages
   <br>

# Algorithms:

1. Dijkstra
2. A\* Search
3. RRT (Rapidly-exploring Random Tree)
4. Depth first search
5. Breadth first search
6. Potential field
