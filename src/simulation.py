import numpy as np
from src.planner import SocialMomentumPlanner

class Robot:
    def __init__(self, start_pos, planner):
        self.robot_state = {"pos": np.array(start_pos, dtype=float), "vel": np.array([0.0, 0.0], dtype=float)}
        self.planner = planner

    def move(self, humans, goal):
        """ Decide next movement based on Social Momentum. """
        best_action = self.planner.optimize_motion(self.robot_state, humans, goal)
        self.robot_state["vel"] = best_action
        self.robot_state["pos"] += best_action  # Move robot

class Human:
    def __init__(self, start_pos, velocity):
        self.pos = np.array(start_pos, dtype=float)
        self.vel = np.array(velocity, dtype=float)

    def move(self):
        """ Move in a straight line (downward in the corridor). """
        self.pos += self.vel