# code structure follows Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

import autograd
import autograd.numpy as np

import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

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

def get_trajectory(t_span=[0,3], timescale=15, radius=None, y0=None, noise_std=0.1, **kwargs):
    t_eval = np.linspace(t_span[0], t_span[1], int(timescale*(t_span[1]-t_span[0])))
    
    # get initial state
    # if y0 is None:
    #     y0 = np.random.rand(2)*2. - 1.
    # if radius is None:
    #     radius = np.random.rand() + 1.3 # sample a range of radii
    # y0 = y0 / np.sqrt((y0**2).sum()) * radius ## set the appropriate radius

    y0 = (np.random.rand(2) - 0.5) * 4

    spring_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=1e-10, **kwargs)
    q, p = spring_ivp['y'][0], spring_ivp['y'][1]
    dydt = [dynamics_fn(None, y) for y in spring_ivp['y'].T]
    dydt = np.stack(dydt).T
    dqdt, dpdt = np.split(dydt,2)
    
    # add noise
    q += np.random.randn(*q.shape)*noise_std
    p += np.random.randn(*p.shape)*noise_std
    return q, p, dqdt, dpdt, t_eval

def get_dataset(seed=0, samples=50, test_split=0.5, **kwargs):
    data = {'meta': locals()}

    # randomly sample inputs
    np.random.seed(seed)
    xs, dxs = [], []
    for s in range(samples):
        x, y, dx, dy, t = get_trajectory(noise_std=0.1, **kwargs)
        xs.append( np.stack( [x, y]).T )
        dxs.append( np.stack( [dx, dy]).T )
        
    data['x'] = np.stack(xs, axis=1) # fit Neural ODE format (45, 50, 2)

    # make a train/test split
    split_ix = int(samples * test_split)
    split_data = {}
    split_data['x'], split_data['test_x'] = data['x'][:,:split_ix,:], data['x'][:,split_ix:,:]

    data = split_data
    data['t'] = t
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