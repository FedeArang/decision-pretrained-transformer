import argparse
import os
import pickle
import random

import gym
import numpy as np
from skimage.transform import resize
from IPython import embed
import jax
import jax.numpy as jnp
import jumanji
from jumanji.environments.routing.maze.types import Position, State

import common_args
from envs import darkroom_env, bandit_env
from ctrls.ctrl_bandit import ThompsonSamplingPolicy
from evals import eval_bandit
from utils import build_maze_data_filename

from jumanji.environments.routing.maze.types import Position, State
from utils_rl import generate_transition_matrix_maze, generate_reward_function_maze, value_iteration, index_to_state, plot_policy, plot_coverage
from functools import partial
from itertools import permutations



def sample_state(state, key, coverage):
    rows = state.walls.shape[0]
    cols = state.walls.shape[1]
        
    if coverage["mode"]:
        target_row = state.target_position.row
        target_col = state.target_position.col
        coverage_mask = jnp.zeros((rows + 2 * coverage["size"], cols + 2 * coverage["size"]), dtype=bool)
        updates = jnp.ones((1 + 2 * coverage["size"], 1 + 2 * coverage["size"]), dtype=bool)
        coverage_mask = jax.lax.dynamic_update_slice(
                                                    coverage_mask, 
                                                    updates,
                                                    (target_row , target_col)
                                                    )
        coverage_mask = jax.lax.slice(coverage_mask, (coverage["size"],coverage["size"]), (rows+coverage["size"],cols+coverage["size"]))
        mask = ~(coverage_mask | state.walls)

    else:
        mask = ~state.walls
    
    state_indices = jax.random.choice(key, jnp.arange(rows * cols), (1,), p=mask.flatten())
    (state_row, state_col) = jnp.divmod(state_indices, cols)

    sampled_position = Position(row=state_row[0], col=state_col[0])
    state.agent_position = sampled_position

    actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left

    action_mask = jnp.array([
        jnp.all(jnp.array([
            0 <= state.agent_position.row + dr,
            state.agent_position.row + dr < rows,
            0 <= state.agent_position.col + dc,
            state.agent_position.col + dc < cols,
            ~state.walls[state.agent_position.row + dr, state.agent_position.col + dc]
        ]))
        for dr, dc in actions
    ], dtype=bool)

    state.action_mask = action_mask

    return state


def step_fn(state, key, coverage):
    key_action, key_state = jax.random.split(key)
    state = sample_state(state, key_state, coverage)
    action = jax.random.randint(key=key_action, minval=0, maxval=num_actions, shape=())
    new_state, timestep = env.step(state, action)
    return new_state, {
                        "state": [state.agent_position],
                        "action": action, 
                        "next_state": [new_state.agent_position],
                        "reward": timestep.reward,
                        "whole_timestep": timestep
                        }

def run_n_steps(state, key, n, coverage):
    random_keys = jax.random.split(key, n)
    partial_step_fn = partial(step_fn, coverage=coverage)
    state, rollout = jax.lax.scan(partial_step_fn, state, random_keys)
    return rollout

def generate_rollouts(key, state, batch_size, rollout_length, coverage):
    keys_rollout = jax.random.split(key, batch_size)
    rollout = jax.vmap(run_n_steps, in_axes=(0, 0, None, None))(state, keys_rollout, rollout_length, coverage)
    return rollout, keys_rollout

def initialize_environments(key_reset, key_goals, batch_size, env, mode):
    if mode == "fixed_walls":
        keys_init = jnp.tile(key_reset, (batch_size, 1))
        state, _ = jax.vmap(env.reset)(keys_init)
        walls = state.walls[0]
        rows = walls.shape[0]
        cols = walls.shape[1]
        state_indices = jax.random.choice(key=key_goals, a=jnp.arange(rows * cols), shape=(batch_size,), p=~walls.flatten())
        (state_row, state_col) = jnp.divmod(state_indices, cols)
        sampled_position = Position(row = state_row, col = state_col)
        state.target_position = sampled_position

    if mode == "variable_walls":
        keys_init = jax.random.split(key_reset, batch_size)
        state, _ = jax.vmap(env.reset)(keys_init)

    if mode == "no_walls":
        keys_init = jax.random.split(key_reset, batch_size)
        state, _ = jax.vmap(env.reset)(keys_init)
        state.walls = jnp.zeros_like(state.walls, dtype=bool)

    return state, keys_init


def extract_data(rollout, batch_size, keys_init, keys_rollout, random_key, category, perm):
    data = []
    num_query_states = 1
    state_sample_keys = jax.random.split(random_key, batch_size)
    test_permutations = perm["test"]
    train_permutations = perm["train"]
    
    for k in range(batch_size):
        walls = rollout["whole_timestep"].observation.walls[k][0]
        target_position = np.array([rollout["whole_timestep"].observation.target_position.row[k][0], rollout["whole_timestep"].observation.target_position.col[k][0]])

        P = generate_transition_matrix_maze(walls)
        r = generate_reward_function_maze(target_position, walls)
        pi_opt = value_iteration(P, r, 0.99, precision=1e-1)
        # plot_policy(policy=pi_opt, grid=walls, title=f"plots/policies/policy_{category}_{k}")

        query_index = jax.random.choice(state_sample_keys[k], jnp.arange(walls.shape[0] * walls.shape[1]), (num_query_states,), p=~walls.flatten())
        query_states = np.array(index_to_state(query_index, walls.shape[1])).T
        optimal_actions = pi_opt[:, query_index].T

        context_actions = np.array(rollout["action"][k])
        context_actions_one_hot = np.zeros((len(context_actions), len(actions)))
        context_actions_one_hot[np.arange(len(context_actions)), context_actions] = 1

        # Pick a permutation based on the category
        if category == "icl":
            perm_index = jax.random.choice(state_sample_keys[k], jnp.arange(len(test_permutations)))
            perm = test_permutations[perm_index]
        else:
            perm_index = jax.random.choice(state_sample_keys[k], jnp.arange(len(train_permutations)))
            perm = train_permutations[perm_index]
        
        # Apply the permutation to the actions
        context_actions_one_hot = np.dot(context_actions_one_hot, perm)
        optimal_actions = np.dot(optimal_actions, perm)

        data.append(
            {
                "query_state": query_states[0],
                "optimal_action": optimal_actions[0],
                "context_actions": context_actions_one_hot,
                "context_states": np.array(jnp.vstack((rollout["state"][0].row[k], rollout["state"][0].col[k]))).T,
                "context_next_states": np.array(jnp.vstack((rollout["next_state"][0].row[k], rollout["next_state"][0].col[k]))).T,
                "context_rewards": np.array(rollout["reward"][k]),
                "goal_state": target_position,
                "env_key": keys_init[k],
                "rollout_key": keys_rollout[k],
            }
        )

        # plot_coverage(data[k]["context_states"], data[k]["context_next_states"], walls, f"plots/coverage/coverage_{category}_{k}")

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    args = vars(parser.parse_args())
    print("Args: ", args)

    env_name = args['env']
    n_envs = args['envs']
    n_eval_envs = args['envs_eval']
    n_hists = args['hists']
    n_samples = args['samples']
    horizon = args['H']
    dim = args['dim']
    var = args['var']
    cov = args['cov']
    env_id_start = args['env_id_start']
    env_id_end = args['env_id_end']
    lin_d = args['lin_d']


    n_train_envs = int(.8 * n_envs)
    n_test_envs = n_envs - n_train_envs

    config = {'n_hists': n_hists, 'n_samples': n_samples}

    if env_name == 'Maze':
        config.update({'rollin_type': 'uniform'})
        env = jumanji.make("Maze-v0")
        num_actions = env.action_spec.num_values

        random_key = jax.random.PRNGKey(1)
        key_reset, key_train, key_icl, key_iwl = jax.random.split(random_key, 4)
        walls = "fixed_walls"
        coverage_bad = {"mode": True, "size": 2}
        coverage_good = {"mode": False}

        actions = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]

        #_____PROPER PERMUTATIONS____
        # permutations_list = list(permutations(actions))
        # permutations_array = np.array(permutations_list)
        # train_permutations = []
        # test_permutation = np.array([actions])
        # for i in range(len(permutations_list)):
        #     if np.all(~np.all(test_permutation == permutations_array[i], axis=1)):
        #         train_permutations.append(permutations_array[i])
        # train_permutations = np.array(train_permutations)

        #_____BAD PERMUTATIONS like in DPT____
        permutations = list(permutations(actions))
        random.shuffle(permutations)
        train_permutations = permutations[:18]
        test_permutation = permutations[18:]

        perm = {"test": test_permutation, "train": train_permutations}

        
        state_train, keys_init_train = initialize_environments(key_reset, key_train, n_train_envs, env, walls)
        state_icl, keys_init_icl = initialize_environments(key_reset, key_icl, n_test_envs, env, walls)
        state_iwl, keys_init_iwl = initialize_environments(key_reset, key_iwl, n_test_envs, env, walls)

        rollout_train, keys_rollout_train = generate_rollouts(key_train, state_train, n_train_envs, horizon, coverage=coverage_good)
        rollout_icl, keys_rollout_icl = generate_rollouts(key_icl, state_icl, n_test_envs, horizon, coverage=coverage_good)
        rollout_iwl, keys_rollout_iwl = generate_rollouts(key_iwl, state_iwl, n_test_envs, horizon, coverage=coverage_bad)

        data_train = extract_data(rollout_train, n_train_envs, keys_init_train, keys_rollout_train, key_train, category="train", perm=perm)
        data_icl = extract_data(rollout_icl, n_test_envs, keys_init_icl, keys_rollout_icl, key_icl, category="icl", perm=perm)
        data_iwl = extract_data(rollout_iwl, n_test_envs, keys_init_iwl, keys_rollout_iwl, key_iwl, category="iwl", perm=perm)

        train_filepath = build_maze_data_filename(
            env_name, n_envs, horizon, config, mode=0, coverage=coverage_good, walls=walls)
        icl_filepath = build_maze_data_filename(
            env_name, n_envs, horizon, config, mode=1, coverage=coverage_good, walls=walls)
        iwl_filepath = build_maze_data_filename(
            env_name, n_envs, horizon, config, mode=2, coverage=coverage_bad, walls = walls)

    else:
        raise NotImplementedError


    if not os.path.exists('datasets'):
        os.makedirs('datasets', exist_ok=True)
    with open(train_filepath, 'wb') as file:
        pickle.dump(data_train, file)
    with open(icl_filepath, 'wb') as file:
        pickle.dump(data_icl, file)
    with open(iwl_filepath, 'wb') as file:
        pickle.dump(data_iwl, file)

    print(f"Saved to {train_filepath}.")
    print(f"Saved to {icl_filepath}.")
    print(f"Saved to {iwl_filepath}.")
