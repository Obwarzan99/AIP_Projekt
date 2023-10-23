import random
import gym
import numpy as np

# Tworzenie monitorowanego środowiska CartPole
env = gym.make('CartPole-v1', render_mode="human")
print()

# Parametry uczenia Q-learning
learning_rate = 0.1
discount_factor = 0.99
exploration_prob = 0.2

# Inicjalizacja tabeli Q
action_space_size = env.action_space.n
state_space_size = env.observation_space.shape[0]
q_table = np.random.rand(2**state_space_size, action_space_size)

# Funkcja konwersji stanu na wartość indeksu
def state_to_index(state):
    return sum(2**i for i, val in enumerate(state))

# Funkcja epsilon-greedy
def epsilon_greedy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Losowa akcja
    else:
        state_index = state_to_index(state)
        return np.argmax(q_table[state_index])

# Uczenie Q-learning
episodes = 10000
for episode in range(episodes):
    state = env.reset()
    done = False
    env.render()  # Włącz renderowanie po wywołaniu env.reset()

    while not done:
        action = epsilon_greedy(state, exploration_prob)
        next_state, reward, done, _ = env.step(action)

        state_index = state_to_index(state)
        next_state_index = state_to_index(next_state)

        # Aktualizacja wartości Q zgodnie z algorytmem Q-learning
        current_q = q_table[state_index][action]
        max_future_q = np.max(q_table[next_state_index])
        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_factor * max_future_q)
        q_table[state_index][action] = new_q

        state = next_state

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}")

# Testowanie nauczonych wartości Q
total_reward = 0
test_episodes = 10
for _ in range(test_episodes):
    state = env.reset()
    done = False

    while not done:
        action = np.argmax(q_table[state_to_index(state)])
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

# Średnia nagroda w testowych epizodach
average_reward = total_reward / test_episodes
print(f"Average Reward: {average_reward}")

# Zakończenie monitorowania
env.close()