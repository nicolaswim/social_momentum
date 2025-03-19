import numpy as np

class SocialMomentumPlanner:
    def __init__(self, avoidance_weight=0.5, goal_weight=0.7):
        self.avoidance_weight = avoidance_weight
        self.goal_weight = goal_weight

    def compute_angular_momentum(self, robot_pos, robot_vel, human_pos, human_vel):
        """ Compute the angular momentum between the robot and a human agent. """
        center_mass = (robot_pos + human_pos) / 2
        p_r = robot_pos - center_mass
        p_h = human_pos - center_mass
        L_rh = np.cross(p_r, robot_vel) + np.cross(p_h, human_vel)  # Angular momentum
        return L_rh

    def optimize_motion(self, robot_state, humans, goal):
        """ Optimize robot motion to balance Social Momentum and goal-seeking. """
        best_action = None
        best_score = float('-inf')

        for action in self.sample_actions():
            # Compute Social Momentum Score
            momentum_score = sum(self.compute_angular_momentum(
                robot_state["pos"], action, h["pos"], h["vel"]) for h in humans)
            
            # Compute Goal-Seeking Score
            direction_to_goal = goal - robot_state["pos"]
            direction_to_goal = direction_to_goal / np.linalg.norm(direction_to_goal)
            goal_score = np.dot(action, direction_to_goal)  # Alignment with goal

            # Weighted sum of momentum and goal-seeking
            total_score = self.avoidance_weight * momentum_score + self.goal_weight * goal_score

            if total_score > best_score:
                best_score = total_score
                best_action = action

        return best_action

    def sample_actions(self):
        """ Generate a set of possible movement actions. """
        step_size = 0.1  # Adjust to change speed
        return [
            np.array([step_size, 0]), np.array([-step_size, 0]),  # Left-Right
            np.array([0, step_size]), np.array([0, -step_size]),  # Up-Down
            np.array([step_size, step_size]), np.array([-step_size, -step_size]),  # Diagonal
            np.array([step_size, -step_size]), np.array([-step_size, step_size])
        ]