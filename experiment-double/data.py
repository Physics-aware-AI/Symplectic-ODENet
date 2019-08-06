# code structure follows Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
from utils import to_pickle, from_pickle
import gym
import myenv


def get_theta(cos, sin):
    theta = np.arctan2(sin, cos)
    theta = theta + 2*np.pi if theta < -np.pi else theta
    theta = theta - 2*np.pi if theta > np.pi else theta
    return theta

def get_q_p(obs):
    '''construct q and p from gym observations of Acrobot-v1'''
    x1 = 0.5 * 1 * obs[1]
    y1 = 0.5 * 1 * obs[0]
    theta_1 = get_theta(obs[0], obs[1])
    x2 = 1 * obs[1] + 0.5 * 1 * (obs[1]*obs[2] + obs[3]*obs[0])
    y2 = 1 * obs[0] + 0.5 * 1 * (obs[0]*obs[2] - obs[1]*obs[3])
    theta_2 = get_theta(obs[2], obs[3])
    x1_dot = 0.5 * 1 * obs[0] * obs[4]
    y1_dot = 0.5 * 1 * (-obs[1]) * obs[4]
    x2_dot = 1 * obs[0] * obs[4] + 0.5 * 1 * (obs[0]*obs[2] - obs[1]*obs[3]) * (obs[4] + obs[5])
    y2_dot = 1 * (-obs[1]) * obs[4] + 0.5 * 1 * (-1) * (obs[1]*obs[2] + obs[3]*obs[0]) * (obs[4] + obs[5])
    return np.array([x1, y1, theta_1, x2, y2, theta_1+theta_2, x1_dot, y1_dot, obs[4]/12, x2_dot, y2_dot, (obs[4] + obs[5])/12])

def sample_gym(seed=0, timesteps=103, trials=200, side=28, min_angle=0., max_angle=np.pi/6, 
              verbose=False, u=0.0, env_name='MyAcrobot-v0'):
    
    gym_settings = locals()
    if verbose:
        print("Making a dataset of Acrobat observations.")
    env: gym.wrappers.time_limit.TimeLimit = gym.make(env_name)
    env.seed(seed)

    trajs = []
    for trial in range(trials):
        valid = False
        while not valid:
            env.reset()
            traj = []
            for step in range(timesteps):
                obs, _, _, _ = env.step(u) # no action
                # x = get_q_p(obs)
                traj.append(obs)
            traj = np.stack(traj)
            if np.amax(traj[:, 4]) < env.MAX_VEL_1 - 0.0001 and np.amin(traj[:, 4]) > -env.MAX_VEL_1 + 0.0001:
                if np.amax(traj[:, 5]) < env.MAX_VEL_2 - 0.0001 and np.amin(traj[:, 5]) > -env.MAX_VEL_2 + 0.0001:
                    valid = True
        trajs.append(traj)
    trajs = np.stack(trajs) # (trials, timesteps, 6)
    trajs = np.transpose(trajs, (1, 0, 2)) # (timesteps, trails, 6)
    tspan = np.arange(timesteps) * env.dt
    return trajs, tspan, gym_settings

def get_dataset(seed=0, samples=50, test_split=0.5, gym=False, save_dir=None, us=[0], rad=False, **kwargs):
    '''Returns a dataset bult on top of OpenAI Gym observations. Also constructs
    the dataset if no saved version is available.'''

    path = '{}/{}-gym-dataset.pkl'.format(save_dir, 'MyAcrobot')
    data = {}
    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except:
        print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
        trajs_force = []
        for u in us:
            trajs, tspan, _ = sample_gym(seed=seed, trials=samples, u=u, **kwargs)
            trajs_force.append(trajs)
        data['x'] = np.stack(trajs_force, axis=0)
        split_ix = int(samples * test_split)
        split_data = {}
        split_data['x'], split_data['test_x'] = data['x'][:,:,:split_ix,:], data['x'][:,:,split_ix:,:]

        data = split_data
        data['t'] = tspan

    return data

def hamiltonian_fn(coords):
    q, p = np.split(coords,2)
    H = 3*(1-np.cos(q)) + p**2 # pendulum hamiltonian
    return H

def dynamics_fn(t, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = np.split(dcoords,2)
    d = 0.1
    S = np.concatenate([dpdt, -dqdt - d * coords[1]], axis=-1)
    return S

def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    field = {'meta': locals()}

    # meshgrid to get vector field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    ys = np.stack([b.flatten(), a.flatten()])
    
    # get vector directions
    dydt = [dynamics_fn(None, y) for y in ys.T]
    dydt = np.stack(dydt).T

    field['x'] = ys.T
    field['dx'] = dydt.T
    return field

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