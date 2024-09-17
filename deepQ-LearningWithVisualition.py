import gymnasium as gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import cv2

class DQLAgent:

    def __init__(self, env):
        # Parameters
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        self.gamma = 0.95
        self.learning_rate = 0.001

        self.epsilon = 1.00
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.memory = deque(maxlen=1000)

        self.model = self.build_model()

        # OpenCV görselleştirme için parametreler
        self.img_height, self.img_width = 800, 1200
        self.img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

    def build_model(self):
        # Neural network for deep Q-Learning
        model = Sequential()
        model.add(Dense(36, activation='tanh', input_shape=(self.state_size,)))
        model.add(Dense(24, activation='tanh'))
        model.add(Dense(18, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # Storage
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Acting
        state = np.array(state).reshape(1, self.state_size)

        if random.uniform(0, 1) <= self.epsilon:
            return env.action_space.sample()
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.array(state).reshape(1, self.state_size)
            next_state = np.array(next_state).reshape(1, self.state_size)

            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state))

            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state, train_target, verbose=0)

    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def visualize_weights_with_opencv(self, current_action):
        # Pull the weights from the layers of the network
        weights = self.model.get_weights()

        # Create a blank image for visualization
        self.img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

        # Calculation of space between layers
        num_layers = len(weights) // 2
        layer_spacing = self.img_width // (num_layers + 2)
        neuron_spacing = self.img_height // (max([w.shape[0] for w in weights[::2]]) + 1)

        # Number of layers and number of neurons in each layer
        layer_sizes = [weights[i].shape[0] for i in range(0, len(weights), 2)]
        layer_sizes.append(weights[-2].shape[1])  # Son katmanın çıkış boyutu

        # Draw layers
        for i, size in enumerate(layer_sizes[:-1]):
            next_size = layer_sizes[i + 1]

            for j in range(size):
                x1, y1 = (i + 1) * layer_spacing, (
                            self.img_height - (size - 1) * neuron_spacing) // 2 + j * neuron_spacing
                color = (0, 255, 0) if i == 0 else (255, 255, 255)
                cv2.circle(self.img, (x1, y1), 15, color, -1)  # Show neurons with colored circles

                # Draw connections for each neuron in the next layer
                for k in range(next_size):
                    x2, y2 = (i + 2) * layer_spacing, (
                                self.img_height - (next_size - 1) * neuron_spacing) // 2 + k * neuron_spacing

                    weight = weights[2 * i][j, k]  # We pull the weights

                    # Weight thickness and color (negative blue, positive red)
                    thickness = max(1, int(2 * np.abs(weight)))  # Minimum 1 for thickness
                    color = (0, 0, 255) if weight > 0 else (255, 0, 0)

                    cv2.line(self.img, (x1, y1), (x2, y2), color, thickness)

        # Draw output neurons
        action_names = ["Left", "Right", "No Action"]
        output_x = (len(layer_sizes)) * layer_spacing
        for j in range(layer_sizes[-1]):
            y1 = (self.img_height - (layer_sizes[-1] - 1) * neuron_spacing) // 2 + j * neuron_spacing
            color = (0, 255, 0) if j == current_action else (128, 128, 128)
            cv2.circle(self.img, (output_x, y1), 15, color, -1)

            cv2.putText(self.img, action_names[j], (output_x + 20, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Neural Network Weights', self.img)
        cv2.waitKey(1)

if __name__ == '__main__':

    # initialize env and agent
    env = gym.make("CartPole-v1", render_mode='human')
    agent = DQLAgent(env)
    batch_size = 10000
    episodes = 1000

    for e in range(episodes):

        # initialize env
        state, _ = env.reset()  # Only take the state, ignore additional info
        state = np.array(state)  # Ensure state is a NumPy array
        state = np.reshape(state, [1, agent.state_size])

        time = 0

        while True:

            # act
            action = agent.act(state)  # select an action

            # Görselleştirme: Ağırlıkları çizdir
            agent.visualize_weights_with_opencv(action)

            # step
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state)  # Ensure next_state is a NumPy array
            next_state = np.reshape(next_state, [1, agent.state_size])

            # remember/storage
            agent.remember(state, action, reward, next_state, done)

            # update state
            state = next_state

            # replay
            agent.replay(batch_size)

            # adjust epsilon
            agent.adaptiveEGreedy()

            time += 1

            if done:
                print("Episode {}, Time {}".format(e, time))
                break

# Test the trained model
import time

# Eğitimli model kullanılarak çevreyi test et
state, _ = env.reset()  # Ignore additional info from reset
state = np.array(state)  # Ensure state is a NumPy array
state = np.reshape(state, [1, agent.state_size])
time_t = 0

while True:

    env.render()

    # Aksiyon seçimi
    action = agent.act(state)

    # Çevre adımı
    next_state, reward, done, _, _ = env.step(action)
    next_state = np.array(next_state)  # Ensure next_state is a NumPy array
    next_state = np.reshape(next_state, [1, agent.state_size])

    # Durumu güncelle
    state = next_state

    time_t += 1
    print(f"Time step: {time_t}")

    if done:
        break

print("Done")
