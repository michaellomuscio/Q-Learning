import numpy as np

class GridWithObstacle:
    def __init__(self):
        # We define a 3x3 grid, which gives us 9 total states.
        # The states are indexed as follows:
        # 0 1 2
        # 3 4 5
        # 6 7 8
        #
        # State 0 is the top-left corner (start).
        # State 8 is the bottom-right corner (goal).
        # State 4 is the obstacle we cannot move onto.
        
        self.rows = 3
        self.cols = 3
        self.n_states = self.rows * self.cols  # 9 states total
        
        # We have 4 possible actions: Up=0, Right=1, Down=2, Left=3
        self.n_actions = 4
        
        # Define the start, goal, and obstacle states.
        self.start_state = 0
        self.goal_state = 8
        self.obstacle_state = 4

    def state_to_rowcol(self, s):
        # Convert a state index into (row, col).
        # Integer division and modulus are used to determine row and column.
        return s // self.cols, s % self.cols

    def rowcol_to_state(self, r, c):
        # Convert (row, col) back into the state index.
        return r * self.cols + c

    def step(self, state, action):
        # Given the current state and an action, this function returns:
        # next_state, reward, done
        #
        # done indicates if we reached the goal.
        # reward is given based on the transition outcome.
        
        # If the agent is already at the goal, we just return immediately.
        if state == self.goal_state:
            return state, 0, True

        # Convert the current state to its (row, col) representation.
        r, c = self.state_to_rowcol(state)
        
        # Determine the next state based on the chosen action.
        # We clamp movements so the agent doesn't leave the grid.
        if action == 0:  # Up
            new_r = max(r - 1, 0)
            new_c = c
        elif action == 1:  # Right
            new_r = r
            new_c = min(c + 1, self.cols - 1)
        elif action == 2:  # Down
            new_r = min(r + 1, self.rows - 1)
            new_c = c
        elif action == 3:  # Left
            new_r = r
            new_c = max(c - 1, 0)

        # Convert the new coordinates back to a state index.
        next_state = self.rowcol_to_state(new_r, new_c)
        
        # Check if the next_state is the obstacle.
        if next_state == self.obstacle_state:
            # If the agent tries to move onto the obstacle:
            # 1) The agent does not move; it stays in the same state.
            # 2) The agent receives a strong negative reward to discourage this action.
            # This negative reward helps the agent learn to avoid attempting to move onto the obstacle.
            next_state = state
            reward = -1.0  # Large negative reward
            done = False
        else:
            # If it's not the obstacle, we give rewards based on reaching the goal or not.
            if next_state == self.goal_state:
                # Reaching the goal yields a positive reward.
                reward = 1.0
                done = True
            else:
                # If not the goal and not an obstacle:
                # We give a small negative step cost (e.g., -0.01).
                # This encourages the agent to find a route to the goal quickly,
                # rather than wandering aimlessly.
                reward = -0.01
                done = False
        
        return next_state, reward, done

    def reset(self):
        # Reset the environment to the start state.
        return self.start_state

# -----------------------------
# Q-LEARNING PARAMETERS
# -----------------------------
#
# alpha (learning rate): Controls how aggressively we update Q-values.
# A common choice is something like 0.1. A higher alpha means we rely more on new information,
# a lower alpha means we trust our existing estimates more.
alpha = 0.1

# gamma (discount factor): Determines how much we value future rewards relative to immediate ones.
# gamma = 0.9 means we consider future rewards important but not as important as immediate ones.
# A gamma close to 1.0 means we value future rewards almost as much as current rewards.
gamma = 0.9

# epsilon (exploration rate): Initially, we set epsilon = 1.0 (100% random at the start)
# This encourages broad exploration at the beginning of training when we know nothing.
# We will decay epsilon over time so that, as we learn more, we rely more on our learned Q-values.
epsilon = 1.0
epsilon_min = 0.01   # The minimum value epsilon can take
epsilon_decay = 0.999 # Each episode, we multiply epsilon by this factor to gradually reduce randomness

# episodes: The number of training episodes.
# Each episode starts at the start state and ends when we reach the goal or
# repeatedly fail to find it until it ends naturally.
# 5000 episodes are used to ensure that we have enough interactions for stable learning,
# especially now that we have negative rewards and decaying epsilon.
episodes = 5000

# Create an instance of our environment.
env = GridWithObstacle()

# Initialize the Q-table:
# Q is a matrix with dimensions [number_of_states x number_of_actions].
# Initially, Q(s,a) = 0 for all s, a because we have no knowledge yet.
Q = np.zeros((env.n_states, env.n_actions))

# -----------------------------
# Q-LEARNING TRAINING LOOP
# -----------------------------
#
# For each episode:
# 1) Reset the environment to the start state.
# 2) Follow the epsilon-greedy policy to select actions until we reach the goal.
# 3) Update Q-values using the Q-learning update rule.
# Over many episodes, the Q-values should converge to values that encourage reaching the goal.

for episode in range(episodes):
    # Start at the beginning of the grid each episode.
    state = env.reset()
    done = False
    
    # Decay epsilon. Over time, epsilon gets smaller, meaning we exploit more and explore less.
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    while not done:
        # Epsilon-greedy action selection:
        # With probability epsilon, we pick a random action (exploration).
        # With probability (1-epsilon), we pick the best action known so far (exploitation).
        if np.random.rand() < epsilon:
            action = np.random.randint(env.n_actions)
        else:
            action = np.argmax(Q[state, :])

        # Take the chosen action in the environment
        next_state, reward, done = env.step(state, action)

        # Q-LEARNING UPDATE:
        # Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a'(Q(s',a')) - Q(s,a)]
        #
        # Interpretation:
        # - We adjust the Q-value of the chosen action towards the observed reward plus the value
        #   of the next state (discounted by gamma).
        # - If this difference is positive, we increase Q(s,a); if negative, we decrease it.
        
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        # Move to the next state.
        state = next_state

# After training completes, we have a Q-table filled with learned values.
# We can derive a policy from the Q-table by picking the action with the highest Q-value in each state.

print("Final Q-Table:")
print(Q)
# The Q-table now contains learned values. Ideally, states around the obstacle will reflect
# that trying to move onto it yields a negative Q-value, thus guiding the agent around it.

# -----------------------------
# DERIVE THE POLICY FROM Q
# -----------------------------
#
# The policy is a mapping from state to best action. We choose the action that has the highest Q-value.
# We'll represent actions with arrows for clarity:
# Up=U, Right=R, Down=D, Left=L
# For the obstacle and goal states, we use special symbols: 'X' for obstacle, 'G' for goal.

actions = ["U","R","D","L"]
policy = []
for s in range(env.n_states):
    if s == env.goal_state:
        # Goal state: We mark it as 'G'
        policy.append("G")
    elif s == env.obstacle_state:
        # Obstacle state: Mark as 'X'
        policy.append("X")
    else:
        # For regular states, pick the action with the highest Q-value.
        best_action = np.argmax(Q[s, :])
        policy.append(actions[best_action])

print("\nDerived Policy:")
for i in range(env.rows):
    # Print the policy row by row.
    row = policy[i*env.cols:(i+1)*env.cols]
    print(row)

# -----------------------------
# COMMENTS ON EXPECTED BEHAVIOR
# -----------------------------
# To deal with the obstacle we added:
# - A heavy penalty (-1.0) for trying to move onto the obstacle discourages the agent from that action.
# - A small negative step cost (-0.01) discourages random wandering and encourages a direct path.
# - A sufficiently large number of episodes (5000) and a decaying epsilon ensures that over time,
#   the agent explores enough in the beginning but later converges to exploiting the best strategy found.
#
# The final policy should reflect a sensible path around the obstacle. For example:
# It might go around the obstacle by moving first right and down, or down and then right,
# avoiding the center cell completely. This setup gives the agent a strong reason not to "try" the obstacle.
