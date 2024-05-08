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


def rollin_mdp(env, rollin_type, optimal_actions, seed = 10):
    states = []
    actions = []
    next_states = []
    rewards = []

    key = jax.random.PRNGKey(seed)

    state, timestep = env.reset(key)

    goal_state = state['target_position']
    walls = state['walls']
    maze_key = state['key']

    for i in range(env.time_limit):
        if rollin_type == 'uniform':
            state = sample_state(env, walls, goal_state, maze_key, i)
            action = sample_action()  
        elif rollin_type == 'expert':
            action = optimal_actions[state] 
        else:
            raise NotImplementedError
        
        next_state, timestep = env.step(state, action)
        reward = timestep['reward']


        agent_position = state['agent_position']  # TODO this is a Position type. which kind of data structure do we want? (E.G. array/list)
        next_agent_position = next_state['agent_position']

        states.append([agent_position[0], agent_position[1]])
        actions.append(action)
        next_states.append([next_agent_position[0], next_agent_position[1]])
        rewards.append(reward)
        state = next_state

    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    rewards = np.array(rewards)

    return states, actions, next_states, rewards, goal_state, walls


def rand_pos_and_dir(env):
    pos_vec = np.random.uniform(0, env.size, size=3)
    pos_vec[1] = 0.0
    dir_vec = np.random.uniform(0, 2 * np.pi)
    return pos_vec, dir_vec


def generate_maze_histories_from_envs(envs, n_hists, n_samples, rollin_type):
    trajs = []
    for env in envs:

        optimal_actions = find_optimal_actions(env) #TODO: since in generate_mdp_histories the optimal actions are required and usually they are part

        for j in range(n_hists):
            (context_states, context_actions, context_next_states, context_rewards, goal_state, walls) = rollin_mdp(env, rollin_type=rollin_type, optimal_actions=optimal_actions)
            
            for k in range(n_samples):
                query_state = sample_state(env, walls) 
                optimal_action = optimal_actions[query_state[0]][query_state[1]]
                
                traj = {
                    'query_state': query_state,
                    'optimal_action': optimal_action,
                    'context_states': context_states,
                    'context_actions': context_actions,
                    'context_next_states': context_next_states,
                    'context_rewards': context_rewards,
                    'goal': goal_state,
                }

                trajs.append(traj)
    return trajs

    
def generate_maze_histories(horizon, n_envs, **kwargs):

    envs = [jumanji.environments.Maze(time_limit = horizon) for _ in range(n_envs)]
    trajs = generate_maze_histories_from_envs(envs, **kwargs)
                                              
    return trajs


def find_optimal_actions(env): #TODO

    #it should be a dictionary where the keys are 'Position' types and the values are the corresponding actions

    #raise NotImplementedError

    return [[sample_action() for j in range(env.num_cols)] for i in range(env.num_rows)]

def sample_state(env, walls, target_position=None, maze_key=None, step_count=None):

    seed = np.random.randint(low = 0, high = env.num_rows*env.num_cols)

    key = jax.random.PRNGKey(seed)
    state_indices = jax.random.choice(key, jnp.arange(env.num_rows * env.num_cols), (1,), replace=False, p=~walls.flatten())
    (state_row, state_col) = jnp.divmod(state_indices, env.num_cols)

    agent_position = Position(row = state_row[0], col = state_col[0])
    
    if target_position is None:
        return agent_position #in this case we only want to return the agent position and not the full state
    else:
        return State(agent_position=agent_position, target_position=target_position, walls=walls, action_mask=jnp.array([True, True, True, True]), key=maze_key, step_count=jnp.array(step_count+1, jnp.int32))



def sample_action():

    return np.random.choice([1, 2, 3, 4])


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)

    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    args = vars(parser.parse_args())
    print("Args: ", args)

    env = args['env']
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

    if env == 'Maze':

        config.update({'rollin_type': 'uniform'})

        train_trajs = generate_maze_histories(horizon, n_train_envs, **config)
        test_trajs = generate_maze_histories(horizon, n_test_envs, **config)
        eval_trajs = generate_maze_histories(horizon, n_eval_envs, **config)

        train_filepath = build_maze_data_filename(
            env, n_envs, dim, horizon, config, mode=0)
        test_filepath = build_maze_data_filename(
            env, n_envs, dim, horizon, config, mode=1)
        eval_filepath = build_maze_data_filename(env, 100, dim, horizon, config, mode=2)



    else:
        raise NotImplementedError


    if not os.path.exists('datasets'):
        os.makedirs('datasets', exist_ok=True)
    with open(train_filepath, 'wb') as file:
        pickle.dump(train_trajs, file)
    with open(test_filepath, 'wb') as file:
        pickle.dump(test_trajs, file)
    with open(eval_filepath, 'wb') as file:
        pickle.dump(eval_trajs, file)

    print(f"Saved to {train_filepath}.")
    print(f"Saved to {test_filepath}.")
    print(f"Saved to {eval_filepath}.")
