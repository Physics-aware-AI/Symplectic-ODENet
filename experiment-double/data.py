# code structure follows Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
from utils import to_pickle, from_pickle

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

def sample_gym(seed=0, timesteps=103, trials=200, side=28, min_angle=0., max_angle=np.pi/6, 
              verbose=False, env_name='Acrobat-v1'):
    
    gym_settings = locals()
    if verbose:
        print("Making a dataset of Acrobat observations.")
    env: gym.wrappers.time_limit.TimeLimit = gym.make(env_name)
    env.reset() ; env.seed(seed)

    for step in range(trials*timesteps):

        if step % timesteps == 0:
            angle_ok = False

            while not angle_ok:
                obs = env.reset()
                theta_init = np.abs(get_theta(obs))
                if verbose:
                    print("\tCalled reset. Max angle= {:.3f}".format(theta_init))
                if theta_init > min_angle and theta_init < max)angle:
                    angle_ok = True
            if verbose:
                print("\tRunning environment...")
        
        

def make_gym_dataset(test_split=0.2, **kwargs):
    '''Constructs a dataset of observations from an OpenAI Gym env'''
    = sample_gym(**kwargs)


def get_dataset(experiment_name, save_dir, **kwargs):
    '''Returns a dataset bult on top of OpenAI Gym observations. Also constructs
    the dataset if no saved version is available.'''

    if experiment_name == "pendulum":
        env_name = "Pendulum-v0"
    elif experiment_name == "acrobot":
        env_name = "Acrobot-v1"
    else:
        assert experiment_name in ['pendulum']

    path = '{}/{}-gym-dataset.pkl'.format(save_dir, experiment_name)

    try:
        data = from_pickle(path)
        print("Successfully loaded data from {}".format(path))
    except:
        print("Had a problem loading data from {}. Rebuilding dataset...".format(path))
        data = make_gym_dataset(**kwargs)
        to_pickle(data,path)

    return data

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
            x_stack.append(x[i:-num_points+i+1,:,:])
        else:
            x_stack.append(x[i:,:,:])
    x_stack = np.stack(x_stack)
    x_stack = np.reshape(x_stack, 
                (num_points, -1, x.shape[2]))
    t_eval = t[0:num_points]
    return x_stack, t_eval