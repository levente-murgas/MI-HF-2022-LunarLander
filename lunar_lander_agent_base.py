import numpy as np
# np.random.seed(0)


# The resolution of the observation space
# The four variables of the observation space, from left to right:
#   0: X component of the vector pointing to the middle of the platform from the lander
#   1: Y component of the vector pointing to the middle of the platform from the lander
#   2: X component of the velocity vector of the lander
#   3: Y component of the velocity vector of the lander
OBSERVATION_SPACE_RESOLUTION = [15, 10, 15, 8]  
ALPHA = 0.1
GAMMA = 0.95


class LunarLanderAgentBase:
    def __init__(self, observation_space, action_space, n_iterations):
        self.observation_space = observation_space
        self.q_table = np.zeros([*OBSERVATION_SPACE_RESOLUTION, len(action_space)])
        self.env_action_space = action_space
        self.n_iterations = n_iterations

        self.epsilon = 0.1
        self.iteration = 0
        self.test = False

        self.last_action = 0
        self.last_state = [0,0,0,0]
        self.START_EPSILON_DECAYING = 1
        self.END_EPSILON_DECAYING = self.n_iterations // 2
        self.epsilon_decay_rate = self.epsilon / (self.END_EPSILON_DECAYING - self.START_EPSILON_DECAYING)

    @staticmethod
    def quantize_state(observation_space, state):
        os = np.array(observation_space)
        quantize_os_window_size = (os[:,1] - os[:,0]) / OBSERVATION_SPACE_RESOLUTION
        quantized_state = ((state - os[:,0]) / quantize_os_window_size) - 1
        return tuple(quantized_state.astype(np.int))


    def epoch_end(self, epoch_reward_sum):
        self.q_table[self.last_state + (self.last_action,)] = epoch_reward_sum
        if self.END_EPSILON_DECAYING >= self.iteration >= self.START_EPSILON_DECAYING:
            self.epsilon -= self.epsilon_decay_rate


    
    def learn(self, old_state, action, new_state, reward):
        new_quantized_state = self.quantize_state(self.observation_space, new_state)
        old_quantized_state = self.quantize_state(self.observation_space, old_state)

        self.last_action = action
        self.last_state = old_quantized_state

        max_future_q = np.max(self.q_table[new_quantized_state])

        current_q = self.q_table[old_quantized_state + (action,)]

        updated_q = current_q + ALPHA * (reward + GAMMA * max_future_q - current_q)
        #updated_q = (1 - ALPHA) * current_q + ALPHA * (reward + GAMMA * max_future_q)

        self.q_table[old_quantized_state + (action,)] = updated_q

    def train_end(self):
        # ... TODO
        # self.q_table = None  # TODO
        self.test = True
