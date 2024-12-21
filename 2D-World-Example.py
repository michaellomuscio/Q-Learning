import numpy as np


class GridWorld:
    def __init__(self, rows=4, cols=4):
        self.rows = rows
        self.cols = cols
        self.n_states = rows * cols
        self.n_actions = 4  # Up=0, Right=1, Down=2, Left=3
        self.terminal_state = self.n_states - 1  # The goal is the last state (15)

    def state_to_rowcol(self, s):
        # Convert a state index (0 to 15) into (row, col)
        return s // self.cols, s % self.cols

    def rowcol_to_state(self, r, c):
        # Convert (row, col) back into a single state index
        return r * self.cols + c

    def step(self, state, action):
        # Given a current state and action, this function returns:
        # next_state, reward, done
        # 'done' indicates if we've reached the terminal (goal) state.

        # Unpack the current state's row and column
        r, c = self.state_to_rowcol(state)

        # If we are already at the terminal state, just return it.
        if state == self.terminal_state:
            return state, 0, True

        # Attempt the movement based on the chosen action.
        # We clamp the values so the agent doesn't go outside the grid.
        if action == 0:  # Up
            r = max(r - 1, 0)
        elif action == 1:  # Right
            c = min(c + 1, self.cols - 1)
        elif action == 2:  # Down
            r = min(r + 1, self.rows - 1)
        elif action == 3:  # Left
            c = max(c - 1, 0)

        # Compute the new state after the move
        next_state = self.rowcol_to_state(r, c)

        # Reward is +1 if we reached the terminal state, else 0.
        reward = 1 if next_state == self.terminal_state else 0

        # 'done' is True if we reached the terminal state
        done = (next_state == self.terminal_state)

        return next_state, reward, done

    def reset(self):
        # Reset environment to the start state.
        # We begin every episode at state 0 (top-left corner).
        return 0


# -----------------------------
# Q-LEARNING SETUP AND TRAINING
# -----------------------------

# Parameters:
alpha = 0.1  # Learning rate: how fast we update Q-values each step
gamma = 0.9  # Discount factor: how strongly we value future rewards
epsilon = 0.1  # Exploration rate: probability of choosing a random action
episodes = 500  # Number of episodes to run for training

env = GridWorld()

# Initialize Q-table:
# Dimensions: [number_of_states x number_of_actions]
# Initially, we know nothing about the values, so set them to 0.
Q = np.zeros((env.n_states, env.n_actions))

# We will run multiple episodes. Each episode is one attempt to start from
# state 0 and reach the terminal state. Over episodes, Q-values should improve.
for episode in range(episodes):
    # Start each episode at state 0.
    state = env.reset()
    done = False

    # Loop until we reach the goal (terminal state).
    while not done:
        # Epsilon-greedy action selection:
        # With probability epsilon, choose a random action to explore.
        # With probability (1 - epsilon), choose the best known action (exploit).
        if np.random.rand() < epsilon:
            # Explore: pick an action at random.
            action = np.random.randint(env.n_actions)
        else:
            # Exploit: pick the action with the highest Q-value for the current state.
            action = np.argmax(Q[state, :])

        # Take the chosen action and observe the next state, reward, and whether we're done.
        next_state, reward, done = env.step(state, action)

        # Q-learning update rule:
        # Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        # This adjusts our Q-value for the chosen action in the current state based on:
        # 1) The immediate reward 'r' obtained.
        # 2) The highest Q-value of the next state 's'' (our best guess of future returns).
        # 3) The current Q(s,a) we are trying to improve.

        # We focus on the difference: (r + gamma * max Q(s',a') - Q(s,a))
        # If this difference is positive, it means we underestimated the Q-value, so we increase Q(s,a).
        # If negative, we overestimated and must decrease it.
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        # Move to the next state.
        state = next_state

# After training is complete, we should have Q-values that reflect good strategies.

print("Final Q-Table after 500 episodes of training:")
print(Q)
# Each row corresponds to a state (0 to 15) and each column to an action.
# The numbers show how valuable each action is in that state.
# Actions: 0=Up, 1=Right, 2=Down, 3=Left.
# A higher number means that action is considered more beneficial by the learned policy.

# Derive the best action (policy) from the Q-table:
# For each state, choose the action with the highest Q-value.
actions = ["U", "R", "D", "L"]
policy = []
for s in range(env.n_states):
    if s == env.terminal_state:
        # Terminal state: no action needed, mark as 'G' for goal.
        policy.append("G")
    else:
        best_a = np.argmax(Q[s, :])
        policy.append(actions[best_a])

# Print the policy as a 4x4 grid:
print("\nDerived Policy from the Q-Table:")
for i in range(env.rows):
    row = policy[i * env.cols:(i + 1) * env.cols]
    print(row)

# -----------------------------
# ADDITIONAL EXPLANATION COMMENTS
# -----------------------------
# In the code above, you see how we implement Q-learning step by step.
#
# Initialization:
# - We started with Q(s,a)=0 for all states and actions. The agent initially knows nothing.
#
# Action Selection (Epsilon-Greedy):
# - epsilon=0.1 means there's a 10% chance of taking a random action.
#   This ensures exploration, allowing the agent to find the goal and not
#   just do what it initially thinks might be best.
#
# Update Rule:
# - We use the Q-learning formula to adjust the Q-values. Over many episodes,
#   these values will start to reflect the true value of each action from each state.
#
# Value Backpropagation:
# - When the agent reaches the goal, it gets a reward of +1.
# - This immediately increases the Q-value of the action taken to reach the goal.
# - In future episodes, states that lead to that 'good action' will get updated too,
#   because max Q(s',a') will no longer be zero. This "goodness" spreads backwards
#   through the state space, helping the agent learn a path that reliably leads to the goal.
#
# After Training:
# - The Q-values in the final Q-table represent the learned estimates of the value of each action.
# - From these Q-values, the best action per state forms the learned policy, shown as arrows (U,R,D,L).
#
# By studying these commented lines and experimenting with parameters (alpha, gamma, epsilon, episodes),
# students can see how changes affect the learning speed, stability, and final policy.
