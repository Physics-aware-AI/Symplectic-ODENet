# code borrowed from Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

import numpy as np
from utils import to_pickle, from_pickle
import gym
import myenv


def sample_gym(seed=0, timesteps=10, trials=50, side=28, min_angle=0., 
              verbose=False, u=[0.0, 0.0], env_name='My_FA_Acrobot-v0'):
    
    gym_settings = locals()
    if verbose:
        print("Making a dataset of Acrobot observations.")
    env: gym.wrappers.time_limit.TimeLimit = gym.make(env_name)
    env.seed(seed)

    trajs = []
    for trial in range(trials):
        valid = False
        while not valid:
            env.reset()
            traj = []
            for step in range(timesteps):
                obs, _, _, _ = env.step(u) # action
                x = np.array([obs[0], obs[2], obs[1], obs[3], obs[4], obs[5], u[0], u[1]])
                traj.append(x)
            traj = np.stack(traj)
            if np.amax(traj[:, 4]) < env.MAX_VEL_1 - 0.001  and np.amin(traj[:, 4]) > -env.MAX_VEL_1 + 0.001:
                if np.amax(traj[:, 5]) < env.MAX_VEL_2 - 0.001  and np.amin(traj[:, 5]) > -env.MAX_VEL_2 + 0.001:
                    valid = True
        trajs.append(traj)
    trajs = np.stack(trajs) # (trials, timesteps, 2)
    trajs = np.transpose(trajs, (1, 0, 2)) # (timesteps, trails, 2)
    tspan = np.arange(timesteps) * 0.05
    return trajs, tspan, gym_settings


def get_dataset(seed=0, samples=50, test_split=0.5, save_dir=None, us=[0], rad=False, **kwargs):
    data = {}

    assert save_dir is not None
    path = '{}/acrobot-gym-dataset.pkl'.format(save_dir)
    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except:
        print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
        trajs_force = []
        for u in us:
            trajs, tspan, _ = sample_gym(seed=seed, trials=samples, u=u, **kwargs)
            trajs_force.append(trajs)
        data['x'] = np.stack(trajs_force, axis=0) # (3, 45, 50, 3)
        # make a train/test split
        split_ix = int(samples * test_split)
        split_data = {}
        split_data['x'], split_data['test_x'] = data['x'][:,:,:split_ix,:], data['x'][:,:,split_ix:,:]

        data = split_data
        data['t'] = tspan

        # to_pickle(data, path)
    return data


def arrange_data(x, t, num_points=2):
    '''Arrange data to feed into neural ODE in small chunks'''
    assert num_points>=2 and num_points<=len(t)
    x_stack = []
    for i in range(num_points):
        if i < num_points-1:
            x_stack.append(x[:, i:-num_points+i+1,:,:])
        else:
            x_stack.append(x[:, i:,:,:])
    x_stack = np.stack(x_stack, axis=1)
    x_stack = np.reshape(x_stack, 
                (x.shape[0], num_points, -1, x.shape[3]))
    t_eval = t[0:num_points]
    return x_stack, t_eval
