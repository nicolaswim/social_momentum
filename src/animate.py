import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.simulation import Robot, Human
from src.planner import SocialMomentumPlanner

# Simulation parameters
robot_start = [0.0, 0.0]
goal = np.array([0.0, 8.0])  # Goal position
human_start = [0.0, 6.0]
human_velocity = [0.0, -0.05]  # Moving downward

# Initialize planner, robot, and human
planner = SocialMomentumPlanner()
robot = Robot(robot_start, planner)
human = Human(human_start, human_velocity)

# Set up the animation
fig, ax = plt.subplots()
ax.set_xlim(-3, 3)
ax.set_ylim(-1, 9)
ax.set_title("Social Momentum Navigation in a Corridor")
ax.plot([-2, -2], [-1, 9], 'k--', linewidth=2)  # Left wall
ax.plot([2, 2], [-1, 9], 'k--', linewidth=2)  # Right wall
robot_dot, = ax.plot([], [], 'ro', markersize=10, label="Robot")
human_dot, = ax.plot([], [], 'bo', markersize=8, linestyle='None', label="Human")
goal_dot, = ax.plot(goal[0], goal[1], 'gx', markersize=10, label="Goal")

# Initialization function
def init():
    robot_dot.set_data([], [])
    human_dot.set_data([], [])
    return robot_dot, human_dot

# Update function
def update(frame):
    robot.move([{"pos": human.pos, "vel": human.vel}], goal)  # Robot avoids human
    human.move()  # Human moves downward

    # Update plot
    robot_dot.set_data([robot.robot_state["pos"][0]], [robot.robot_state["pos"][1]])
    human_dot.set_data([human.pos[0]], [human.pos[1]])
    return robot_dot, human_dot

# Create animation
ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=True, interval=100)
plt.legend()
plt.show()