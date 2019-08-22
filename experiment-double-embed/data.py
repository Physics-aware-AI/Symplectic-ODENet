# code borrowed from Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
from utils import to_pickle, from_pickle
import gym
import myenv

def hamiltonian_fn(coords):
    q, p = np.split(coords,2)
    # pendulum hamiltonian conosistent with openAI gym Pendulum-v0
    H = 5*(1-np.cos(q)) + 1.5 * p**2 
    return H

def dynamics_fn(t, coords, u=0):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = np.split(dcoords,2)
    S = np.concatenate([dpdt, -dqdt + u], axis=-1)
    return S

def get_trajectory(t_span=[0,3], timescale=20, radius=None, y0=None, noise_std=0.1, u=0.0, rad=False, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    
    # get initial state
    if rad:
        if y0 is None:
            y0 = np.random.rand(2)*2.-1
        if radius is None:
            radius = np.random.rand() + 1.3 # sample a range of radii
        y0 = y0 / np.sqrt((y0**2).sum()) * radius ## set the appropriate radius
    else:
        if y0 is None:
            y0 = np.random.rand(2) * 3 * np.pi - np.pi

    spring_ivp = solve_ivp(lambda t, y: dynamics_fn(t, y, u), t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q, p = spring_ivp['y'][0], spring_ivp['y'][1]
    # dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
    # dydt = np.stack(dydt).T
    # dqdt, dpdt = np.split(dydt,2)

    # add noise
    q += np.random.randn(*q.shape)*noise_std
    p += np.random.randn(*p.shape)*noise_std

    return q, p, t_eval

def sample_gym(seed=0, timesteps=10, trials=50, side=28, min_angle=0., 
              verbose=False, u=0.0, env_name='MyAcrobot-v0'):
    
    gym_settings = locals()
    if verbose:
        print("Making a dataset of Pendulum observations.")
    env: gym.wrappers.time_limit.TimeLimit = gym.make(env_name)
    env.seed(seed)

    trajs = []
    for trial in range(trials):
        valid = False
        while not valid:
            env.reset()
            traj = []
            for step in range(timesteps):
                obs, _, _, _ = env.step([u]) # action
                x = np.array([obs[0], obs[2], obs[1], obs[3], obs[4], obs[5], u])
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
    path = '{}/pendulum-gym-dataset.pkl'.format(save_dir)
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

def get_field(xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20, u=0):
    field = {'meta': locals()}

    # meshgrid to get vector field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    ys = np.stack([b.flatten(), a.flatten()])
    
    # get vector directions
    dydt = [dynamics_fn(None, y, u) for y in ys.T]
    dydt = np.stack(dydt).T

    field['x'] = ys.T
    field['dx'] = dydt.T
    return field