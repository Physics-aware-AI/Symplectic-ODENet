# code borrowed from Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
from utils import to_pickle, from_pickle
import gym

def hamiltonian_fn(coords):
    q, p = np.split(coords,2)
    # pendulum hamiltonian conosistent with openAI gym Pendulum-v0
    H = 5*(1-np.cos(q)) + 1.5 * p**2 
    return H

def hamiltonian_6d_fn(coords):
    x, y, theta, p_x, p_y, p_theta = np.split(coords, 6, axis=1)
    H = p_x * p_x /2 + p_y * p_y /2 + 6 * p_theta * p_theta + 10 * y + 5
    return H

def dynamics_fn(t, coords):
    dcoords = autograd.grad(hamiltonian_fn)(coords)
    dqdt, dpdt = np.split(dcoords,2)
    S = np.concatenate([dpdt, -dqdt], axis=-1)
    return S

def get_trajectory(t_span=[0,3], timescale=20, radius=None, y0=None, noise_std=0.1, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    
    # # get initial state
    # if y0 is None:
    #     y0 = np.random.rand(2)*2.-1
    # if radius is None:
    #     radius = np.random.rand() + 1.3 # sample a range of radii
    # y0 = y0 / np.sqrt((y0**2).sum()) * radius ## set the appropriate radius

    y0 = (np.random.rand(2) - 0.5) * 5
    
    spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q, p = spring_ivp['y'][0], spring_ivp['y'][1]
    dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
    dydt = np.stack(dydt).T
    dqdt, dpdt = np.split(dydt,2)
    
    # add noise
    q += np.random.randn(*q.shape)*noise_std
    p += np.random.randn(*p.shape)*noise_std
    return q, p, dqdt, dpdt, t_eval

def transform_q_p(q, p):
    '''transform 2D q and p to 6D q and p, vectorized operation'''
    x = 0.5 * np.sin(q)
    y = - 0.5 * np.cos(q)
    theta = q
    p_x = 0.5 * np.cos(q) * 3 * p
    p_y = 0.5 * np.sin(q) * 3 * p
    p_theta = p/4.0 # p_theta in the COM
    return np.stack([x, y, theta, p_x, p_y, p_theta], axis=1)

def get_theta(cos, sin):
    theta = np.arctan2(sin, cos)
    theta = theta + 2*np.pi if theta < -np.pi else theta
    theta = theta - 2*np.pi if theta > np.pi else theta
    return theta


def get_q_p(obs):
    '''construct q and p from gym observations of Pendulum-v0'''
    x = - 0.5 * obs[1]
    y = 0.5 * obs[0]
    theta = get_theta(-obs[0], -obs[1])
    p_x = - 0.5 * obs[0] * obs[2]
    p_y = - 0.5 * obs[1] * obs[2]
    p_theta = obs[2] / 12.0  # p_theta in the COM
    return np.array([x, y, theta, p_x, p_y, p_theta])


def sample_gym(seed=0, timesteps=103, trials=200, side=28, min_angle=0., max_angle=np.pi/6, 
              verbose=False, env_name='Pendulum-v0'):
    
    gym_settings = locals()
    if verbose:
        print("Making a dataset of Pendulum observations.")
    env: gym.wrappers.time_limit.TimeLimit = gym.make(env_name)
    env.reset() ; env.seed(seed)

    trajs = []
    for trial in range(trials):
        traj = []
        for step in range(timesteps):

            if step == 0:
                angle_ok = False

                while not angle_ok:
                    obs = env.reset()
                    theta_init = np.abs(get_theta(-obs[0], -obs[1]))
                    if verbose:
                        print("\tCalled reset. Max angle= {:.3f}".format(theta_init))
                    if theta_init > min_angle and theta_init < max_angle:
                        angle_ok = True
                if verbose:
                    print("\tRunning environment...")
            
            obs, _, _, _ = env.step([0.0]) # no action
            x = get_q_p(obs)
            traj.append(x)
        traj = np.stack(traj)
        trajs.append(traj)
    trajs = np.stack(trajs) # (trials, timesteps, 6)
    trajs = np.transpose(trajs, (1, 0, 2)) # (timesteps, trails, 6)
    tspan = np.arange(timesteps) * 0.05
    return trajs, tspan, gym_settings

def get_dataset(seed=0, samples=50, test_split=0.5, gym=False, save_dir=None, **kwargs):
    data = {}

    if gym:
        assert save_dir is not None
        path = '{}/acrobot-gym-dataset.pkl'.format(save_dir)
        try:
            data = from_pickle(path)
            print("Successfully loaded data from {}".format(path))
        except:
            print("Had a problem loading data from {}. Rebuilding dataset...".format(path))

            trajs, tspan, _ = sample_gym(seed=seed, trials=samples, **kwargs)
            split_ix = int(trajs.shape[1]*test_split)
            data['x'], data['test_x'] = trajs[:, split_ix:, :], trajs[:, :split_ix, :]
            data['t'] = tspan

            to_pickle(data, path)
    else:
        # randomly sample inputs
        np.random.seed(seed)
        xs = []
        for _ in range(samples):
            q, p, _, _, t = get_trajectory(noise_std=0.0, **kwargs)
            q_p_6D = transform_q_p(q, p)
            xs.append(q_p_6D)
            
        data['x'] = np.stack(xs, axis=1) # fit Neural ODE format (45, 50, 2)

        # make a train/test split
        split_ix = int(samples * test_split)
        split_data = {}
        split_data['x'], split_data['test_x'] = data['x'][:,:split_ix,:], data['x'][:,split_ix:,:]

        data = split_data
        data['t'] = t
    
    return data

def arrange_data(x, t, num_points=2, output_diff=False):
    '''Arrange data to feed into neural ODE in small chunks'''
    x_dot = (x[1:,:,:] - x[:-1,:,:])/ (t[1] - t[0])
    x = x[:-1,:,:]
    assert num_points>=2 and num_points<=len(t)
    x_stack = []
    x_dot_stack = []
    for i in range(num_points):
        if i < num_points-1:
            x_stack.append(x[i:-num_points+i+1,:,:])
            x_dot_stack.append(x[i:-num_points+i+1,:,:])
        else:
            x_stack.append(x[i:,:,:])
            x_dot_stack.append(x[i:,:,:])
    x_stack = np.stack(x_stack)
    x_dot_stack = np.stack(x_dot_stack)
    x_stack = np.reshape(x_stack, 
                (num_points, -1, x.shape[2]))
    x_dot_stack = np.reshape(x_dot_stack, 
                (num_points, -1, x_dot.shape[2]))
    t_eval = t[0:num_points]
    if output_diff:
        return x_stack, x_dot_stack, t_eval
    else:
        return x_stack, t_eval