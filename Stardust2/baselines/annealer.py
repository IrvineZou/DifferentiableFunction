from utils import *


class GymAnnealer(Annealer):
    def __init__(self, initial_solution, env, topological_order):
        self.env = env
        self.topological_order = topological_order  # List of nodes in topological order
        super(GymAnnealer, self).__init__(initial_solution)
        self.min_time = np.inf
        self.sample = 0
        self.results = {}

    def move(self):
        # Randomly select a node to modify its action
        node = random.choice(self.topological_order)
        # Update the action for the selected node
        self.state[node] = self.env.action_space.sample()

    def energy(self):
        total_reward = 0
        obs = self.env.reset(True)
        done = False

        # Iterate over nodes in topological order
        for node in self.topological_order:
            # Get the action for the current node
            action = self.state[node]
            # Step the environment with the action
            obs, reward, done, _, info = self.env.step2(action,False)
            total_reward += reward
            if done:
                #if self.env.time < self.min_time:
                #    self.min_time = self.env.time
                #    self.results[self.sample] = self.min_time
                self.sample += 1
                break
        print("total throughput is: ", total_reward)
        return total_reward
