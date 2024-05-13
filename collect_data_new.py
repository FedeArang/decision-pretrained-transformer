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
from utils_rl import generate_transition_matrix_maze, generate_reward_function_maze, value_iteration, index_to_state, plot_policy


def sample_state(state, key):
    rows = state.walls.shape[0]
    cols = state.walls.shape[1]
    state_indices = jax.random.choice(key, jnp.arange(rows * cols), (1,), p=~state.walls.flatten())
    (state_row, state_col) = jnp.divmod(state_indices, cols)
    sampled_position = Position(row = state_row[0], col = state_col[0])
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

def step_fn(state, key):
    action = jax.random.randint(key=key, minval=0, maxval=num_actions, shape=())
    new_state, timestep = env.step(state, action)
    return new_state, {
                        "state": [state.agent_position],
                        "action": action, 
                        "next_state": [new_state.agent_position],
                        "reward": timestep.reward,
                        "whole_timestep": timestep
                        }

def uniform_step_fn(state, key):
    key_action, key_state = jax.random.split(key)
    state = sample_state(state, key_state)
    return step_fn(state, key_action)

def run_n_steps(state, key, n):
    random_keys = jax.random.split(key, n)
    state, rollout = jax.lax.scan(uniform_step_fn, state, random_keys)
    return rollout

def initialize_environments(key, batch_size, env, mode):
    
    if mode == "fixed_walls":
        keys_init = jnp.tile(key, (batch_size, 1))
        state, _ = jax.vmap(env.reset)(keys_init)
        walls = state.walls[0]
        rows = walls.shape[0]
        cols = walls.shape[1]
        state_indices = jax.random.choice(key=key, a=jnp.arange(rows * cols), shape=(batch_size,), p=~walls.flatten())
        (state_row, state_col) = jnp.divmod(state_indices, cols)
        sampled_position = Position(row = state_row, col = state_col)
        state.target_position = sampled_position

    if mode == "variable_walls":
        keys_init = jax.random.split(key, batch_size)
        state, timestep = jax.vmap(env.reset)(keys_init)

    return state, keys_init

def generate_rollouts(key, env, batch_size, rollout_length, mode):
    key1, key2 = jax.random.split(key)
    state, keys_init = initialize_environments(key1, batch_size, env, mode)
    keys_rollout = jax.random.split(key2, batch_size)
    rollout = jax.vmap(run_n_steps, in_axes=(0, 0, None))(state, keys_rollout, rollout_length)
    return rollout, keys_init, keys_rollout

def extract_data(rollout, batch_size, keys_init, keys_rollout, random_key):
    data = []
    num_query_states = 1
    state_sample_keys = jax.random.split(random_key, batch_size)
    for k in range(batch_size):
        walls = rollout["whole_timestep"].observation.walls[k][0]
        target_position = np.array([rollout["whole_timestep"].observation.target_position.row[k][0], rollout["whole_timestep"].observation.target_position.col[k][0]])

        P = generate_transition_matrix_maze(walls)
        r = generate_reward_function_maze(target_position, walls)
        pi_opt = value_iteration(P, r, 0.99, precision=1e-1)
        plot_policy(policy=pi_opt, grid=walls, title=f"policies/policy_{k}")

        query_index =  jax.random.choice(state_sample_keys[k], jnp.arange(walls.shape[0] * walls.shape[1]), (num_query_states,), p=~walls.flatten())
        query_states = np.array(index_to_state(query_index, walls.shape[1])).T
        optimal_actions = pi_opt[:, query_index].T

        context_actions = np.array(rollout["action"][k])
        context_actions_one_hot = np.zeros((len(context_actions), num_actions))
        context_actions_one_hot[np.arange(len(context_actions)), context_actions] = 1

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
        key_train, key_test, key_eval = jax.random.split(random_key, 3)
        mode = "fixed_walls"

        rollout_train, keys_init_train, keys_rollout_train = generate_rollouts(key_train, env, n_train_envs, horizon, mode)
        rollout_test, keys_init_test, keys_rollout_test = generate_rollouts(key_test, env, n_test_envs, horizon, mode)
        rollout_eval, keys_init_eval, keys_rollout_eval = generate_rollouts(key_eval, env, n_eval_envs, horizon, mode)

        data_train = extract_data(rollout_train, n_train_envs, keys_init_train, keys_rollout_train, key_train)
        data_test = extract_data(rollout_test, n_test_envs, keys_init_test, keys_rollout_test, key_test)
        data_eval = extract_data(rollout_eval, n_eval_envs, keys_init_eval, keys_rollout_eval, key_eval)

        print(data_train[0])

        train_filepath = build_maze_data_filename(
            env_name, n_envs, horizon, config, mode=0)
        test_filepath = build_maze_data_filename(
            env_name, n_envs, horizon, config, mode=1)
        eval_filepath = build_maze_data_filename(
            env_name, n_envs, horizon, config, mode=2)

    else:
        raise NotImplementedError


    if not os.path.exists('datasets'):
        os.makedirs('datasets', exist_ok=True)
    with open(train_filepath, 'wb') as file:
        pickle.dump(data_train, file)
    with open(test_filepath, 'wb') as file:
        pickle.dump(data_test, file)
    with open(eval_filepath, 'wb') as file:
        pickle.dump(data_eval, file)

    print(f"Saved to {train_filepath}.")
    print(f"Saved to {test_filepath}.")
    print(f"Saved to {eval_filepath}.")
