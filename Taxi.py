import gym
import numpy as np
import time

# Inicjacja środowiska gry Taxi
env = gym.make('Taxi-v3', render_mode="human")

# Parametry algorytmu Q-learning
gamma = 0.99  # Współczynnik dyskontowania
alpha = 0.1  # Współczynnik uczenia
max_epsilon = 1.0  # Maksymalny współczynnik eksploracji
min_epsilon = 0.01  # Minimalny współczynnik eksploracji
epsilon = 0.01  # Współczynnik eksploracji
decay_rate = 0.9  # Współczynnik spadku eksploracji

counter = 0

# Liczba stanów i akcji w środowisku
num_states = env.observation_space.n
num_actions = env.action_space.n

# Inicjacja tablicy Q, przechowującej wartości funkcji wartości Q
Q = np.zeros((num_states, num_actions))
Q = np.load(file="Q_table.npy")  # Wczytanie wyników uczenia (jeśli dostępne)

# Liczba epizodów uczenia
num_episodes = 5000

# Maksymalna liczba kroków w epizodzie
max_steps_per_episode = 30

# Pętla uczenia
for episode in range(num_episodes):
    counter = counter + 1

    # Resetowanie środowiska do stanu początkowego
    state = env.reset()[0]

    for step in range(max_steps_per_episode):
        # Wybór akcji na podstawie strategii eksploracji (epsilon-greedy)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Losowa akcja
        else:
            action = np.argmax(Q[state, :])

        # Wykonanie wybranej akcji w środowisku
        new_state, reward, done, _, info = env.step(action)

        # Aktualizacja wartości funkcji wartości Q
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

        state = new_state

        if done:
            break


    # Aktualizacja współczynnika eksploracji
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    print("Epsilon:", epsilon , "Episode :" , episode , "Reward :",reward)

#np.save(file="Q_table.npy" ,arr=Q)  # Opcjonalne zapisanie wyników uczenia
print()

# Ocena na podstawie wytrenowanego modelu
total_reward = 0
num_eval_episodes = 100
max_eval_steps_per_episode = 1000

for episode in range(num_eval_episodes):
    state = env.reset()[0]

    for step in range(max_eval_steps_per_episode):
        action = np.argmax(Q[state, :])
        state, reward, done, _, info = env.step(action)
        total_reward += reward

        if done:
            break

mean_reward = total_reward / num_eval_episodes
print(f"Średnia nagroda: {mean_reward:.2f}")

# Wyłącz rendering po zakończeniu oceny
env.close()