import numpy as np
import random

# ==========================================================
# Q-Learning Example with Detailed Comments
#
# In this example, we have a simple 1-D world:
# States: 0 (start), 1, 2, and 3 (goal).
# Actions: 0 = move left, 1 = move right.
#
# The agent starts at state 0 and wants to reach state 3 to get a reward of +1.
# Moving left at state 0 yields -1, all other moves (except reaching the goal) give 0.
#
# Q-Learning will update the Q-values Q[s,a] over multiple episodes.
# After training, the agent should learn to always move to the right until it reaches the goal.
# ==========================================================

# For reproducibility, fix the random seed
np.random.seed(42)
random.seed(42)

# Define state and action space sizes
num_states = 4   # States: 0, 1, 2, 3
num_actions = 2  # Actions: 0 (left), 1 (right)

# Initialize the Q-table with zeros.
# Q is a 2D array with shape [num_states, num_actions].
# Initially, Q[s,a] = 0 for all s and a.
Q = np.zeros((num_states, num_actions))

# ----------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------
alpha = 0.1     # Learning rate: how quickly Q-values are updated after each experience.
gamma = 0.9     # Discount factor: how much future rewards are considered.
epsilon = 0.1   # Exploration rate: probability of choosing a random action over the greedy one.
episodes = 500  # Number of training episodes: how many times the agent attempts to solve the task.

# ----------------------------------------------------------
# Environment dynamics (the "step" function)
# Given a state and an action, this function returns the next state and reward.
# ----------------------------------------------------------
def step(state, action):
    # If we are already at the goal (state 3), staying there yields no additional reward.
    if state == 3:
        return 3, 0.0

    if action == 0:  # Move left
        if state == 0:
            # Can't move left from state 0, penalize to discourage staying stuck here.
            next_state = 0
            reward = -1.0
        else:
            # Generally, moving left just takes us one state back with no immediate reward.
            next_state = state - 1
            reward = 0.0
    else:  # Move right
        # Moving right transitions the agent to the next state.
        next_state = min(state + 1, 3)
        if next_state == 3:
            # Reaching the goal gives a reward of +1.
            reward = 1.0
        else:
            # Otherwise, no immediate reward, but potential future reward if we continue.
            reward = 0.0

    return next_state, reward

# ----------------------------------------------------------
# Action selection function using ε-greedy policy
# With probability ε, take a random action (explore),
# and with probability 1-ε, take the best known action according to Q.
# ----------------------------------------------------------
def choose_action(state, Q, epsilon):
    # Generate a random number to decide whether to explore or exploit.
    if random.uniform(0, 1) < epsilon:
        # Explore: choose a random action
        return random.randint(0, num_actions-1)
    else:
        # Exploit: choose the action with the highest Q-value in this state
        return np.argmax(Q[state, :])

# ----------------------------------------------------------
# Training Loop
#
# We run multiple episodes. Each episode:
# 1. Start from state 0.
# 2. Until we reach the goal (state 3):
#    - Choose an action using ε-greedy.
#    - Perform the action, get next_state and reward.
#    - Update Q[s,a] using the Q-Learning update rule.
#
# Over many episodes, Q-values should converge so that the agent learns the optimal strategy:
# always move right to reach the goal.
# ----------------------------------------------------------
for ep in range(episodes):
    # Start each episode from the start state.
    state = 0
    done = False

    # Keep taking actions until the agent reaches the goal state.
    while not done:
        # 1. Choose an action based on our current Q-values and ε.
        #    If Q-values are not fully learned yet, we need some randomness (ε) to discover better actions.
        action = choose_action(state, Q, epsilon)

        # 2. Take the chosen action and observe the outcome.
        next_state, reward = step(state, action)

        # 3. Q-Learning update rule:
        #    Q(s,a) ← Q(s,a) + α [ r + γ max_{a'}Q(s',a') - Q(s,a) ]
        old_value = Q[state, action]
        # Check the future value: max of Q at next_state
        next_max = np.max(Q[next_state, :])

        # Compute the TD (Temporal Difference) error:
        # reward + discounted future value - old value
        td_error = reward + gamma * next_max - old_value

        # Update Q-value:
        Q[state, action] = old_value + alpha * td_error

        # 4. Move to the next state
        state = next_state

        # If we reached the goal state (3), the episode ends.
        if state == 3:
            done = True

# ----------------------------------------------------------
# After training, let's inspect the learned Q-values.
# We expect Q-values for actions that move towards the goal (right) to be higher
# and those that move away or waste time (left at start) to be lower.
# ----------------------------------------------------------

print("Learned Q-values after training:")
for s in range(num_states):
    print(f"State {s}: Left={Q[s,0]:.4f}, Right={Q[s,1]:.4f}")

# Let's derive the policy from the learned Q-values:
# The best action at each state is the action with the highest Q-value.
actions = ["Left", "Right"]
print("\nDerived Greedy Policy:")
for s in range(num_states):
    best_a = np.argmax(Q[s, :])
    print(f"State {s}: {actions[best_a]}")

# ----------------------------------------------------------
# Explanation Recap (integrated into comments):
#
# - Q-values interpret what future rewards the agent can obtain from each state-action pair.
# - Initially all Q = 0, meaning the agent has no clue about which actions lead to better outcomes.
# - As the agent interacts, it discovers that moving right eventually leads to a reward (+1 at the goal).
# - By using the update rule with a learning rate (α=0.1), Q-values are adjusted gradually.
# - The discount factor (γ=0.9) encourages planning ahead, valuing future rewards.
# - The exploration rate (ε=0.1) ensures that the agent tries different actions enough times to
#   find the better ones. Without exploration, it could get stuck doing something suboptimal.
# - Over 500 episodes, the agent refines its Q-table. Actions that lead towards the goal get higher Q-values.
# - Thus, from state 0, Q(0,right) > Q(0,left). Similarly, from states 1 and 2, moving right is learned to be the best action.
# - The final policy is: always move right until the goal is reached.
#
# By examining the Q-values and derived policy, students see how Q-Learning finds the optimal strategy.
# ----------------------------------------------------------
