import numpy as np
import matplotlib.pyplot as plt

def state_to_index(x, y, columns):
        return x * columns + y

def index_to_state(idx, columns):
    x = idx // columns
    y = idx % columns
    return x, y

def generate_transition_matrix_maze(walls):
    rows = walls.shape[0]  # Number of rows
    columns = walls.shape[1]  # Number of columns
    num_states = rows * columns
    num_actions = 4  # up, right, down, left

    # Transition matrix P(s'|s,a)
    P = np.zeros((num_states, num_states, num_actions))

    # Actions: up (0), right (1), down (2), left (3)
    actions = {
        0: (-1, 0),  # Up
        1: (0, 1),   # Right
        2: (1, 0),   # Down
        3: (0, -1)   # Left
    }

    # Populate the transition matrix
    for x in range(rows):
        for y in range(columns):
            current_state = state_to_index(x, y, columns)
            for action, (dx, dy) in actions.items():
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < rows and 0 <= new_y < columns and not walls[new_x, new_y]:
                    next_state = state_to_index(new_x, new_y, columns)
                else:
                    next_state = current_state  # Stay in place on wall or out-of-bounds
                
                P[next_state, current_state, action] = 1

    return P

def generate_reward_function_maze(target_position, walls):
    rows = walls.shape[0]  # Number of rows
    columns = walls.shape[1]  # Number of columns
    num_states = rows * columns
    num_actions = 4  # up, right, down, left
    rewards = np.zeros(rows * columns)
    target_index = state_to_index(target_position[0], target_position[1], columns)
    rewards[target_index] = 1
    return rewards

def value_iteration(P, reward, discount, precision=1e-2):
    state_size = P.shape[0]
    action_size = P.shape[2]
    value = np.zeros(state_size)
    prev_value = np.ones(state_size)
    pi_vi = np.zeros((action_size, state_size))
    while np.max(np.abs(value - prev_value)) > precision:
        prev_value = value.copy()
        for state in range(state_size):
            value[state] = np.max(
                [
                    reward[state] + discount * np.sum(P[:, state, action] * value)
                    for action in range(action_size)
                ]
            )

    for state in range(state_size):
        values = np.array(
            [
                reward[state] + discount * np.sum(P[:, state, action] * value)
                for action in range(action_size)
            ]
        )

        best_action = np.argmax([reward[state] + discount * np.sum(P[:,state,action] * value) for action in range(action_size)])
        pi_vi[best_action, state] = 1

    return pi_vi

def plot_policy(policy, grid, title):
    n, m = grid.shape
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid, cmap='gray')

    for i in range(n):
        for j in range(m):
            if grid[i, j]:
                continue  # Skip walls
            state = i * m + j
            if policy[0, state] > 0:  # Up
                # print("up")
                ax.arrow(j, i, 0, -0.5, head_width=0.2, head_length=0.2, fc='red', ec='red')
            if policy[1, state] > 0:  # Right
                ax.arrow(j, i, 0.5, 0, head_width=0.2, head_length=0.2, fc='green', ec='green')
                # print("right")
            if policy[2, state] > 0:  # Down
                ax.arrow(j, i, 0, 0.5, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
                # print("down")
            if policy[3, state] > 0:  # Left
                # print("left")
                ax.arrow(j, i, -0.5, 0, head_width=0.2, head_length=0.2, fc='yellow', ec='yellow')
    
    ax.set_xticks(np.arange(-0.5, m, 1))
    ax.set_yticks(np.arange(-0.5, n, 1))
    plt.savefig(title)

def plot_value_function(value, grid, title):
    n, m = grid.shape
    reshaped_value = value.reshape(n, m)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    # Create a heatmap for the value function
    cax = ax.matshow(reshaped_value, cmap='viridis')
    
    # Overlay the grid layout to identify walls
    for i in range(n):
        for j in range(m):
            if grid[i, j] == 1:  # Assuming 1 indicates a wall
                ax.text(j, i, 'W', va='center', ha='center', color='white', fontsize=12)
            else:
                ax.text(j, i, f'{reshaped_value[i, j]:.2f}', va='center', ha='center', color='white', fontsize=8)
    
    # Set up the axis
    ax.set_xticks(np.arange(-0.5, m, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    ax.set_xticks(np.arange(m))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.colorbar(cax)
    plt.savefig(title)