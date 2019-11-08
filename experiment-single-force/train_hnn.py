# This script is for the ablation study of differentiable ODE solver
# modified from https://github.com/greydanus/hamiltonian-nn/blob/master/experiment-pend/train.py


import torch, argparse
import numpy as np

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from nn_models import MLP
from hnn import HNN
from data import get_dataset
from utils import L2_loss

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_dim', default=2, type=int, help='dimensionality of input tensor')
    parser.add_argument('--hidden_dim', default=400, type=int, help='hidden dimension of mlp')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=1000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='pend', type=str, help='only one option right now')
    parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--use_rk4', dest='use_rk4', action='store_true', help='integrate derivative with RK4')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--field_type', default='solenoidal', type=str, help='type of vector field to learn')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--rad', dest='rad', action='store_true', help='generate random data around a radius')
    parser.set_defaults(feature=True)
    return parser.parse_args()

def train(args):
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # init model and optimizer
    if args.verbose:
        print("Training baseline model:" if args.baseline else "Training HNN model:")

    output_dim = args.input_dim if args.baseline else 2
    nn_model = MLP(args.input_dim, 400, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=args.baseline)
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-4)

    # the data API is different
    # make sure it is a fair comparison
    # generate the data the same way as in the SymODEN
    # compute the time derivative based on the generated data
    us = [0.0]
    data = get_dataset(seed=args.seed, save_dir=args.save_dir, 
                    rad=args.rad, us=us, samples=50, timesteps=45) 
    
    # arrange data 

    train_x, t_eval = data['x'][0,:,:,0:2], data['t']
    test_x, t_eval = data['test_x'][0,:,:,0:2], data['t']

    train_dxdt = (train_x[1:,:,:] - train_x[:-1,:,:]) / (t_eval[1] - t_eval[0])
    test_dxdt = (test_x[1:,:,:] - test_x[:-1,:,:]) / (t_eval[1] - t_eval[0])

    train_x = train_x[0:-1,:,:].reshape((-1,2))
    test_x = test_x[0:-1,:,:].reshape((-1,2))
    test_dxdt = test_dxdt.reshape((-1,2))
    train_dxdt = train_dxdt.reshape((-1,2))
    
    x = torch.tensor( train_x, requires_grad=True, dtype=torch.float32)
    test_x = torch.tensor( test_x, requires_grad=True, dtype=torch.float32)
    dxdt = torch.Tensor(train_dxdt)
    test_dxdt = torch.Tensor(test_dxdt)

    # vanilla train loop
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(args.total_steps+1):
        
        # train step
        dxdt_hat = model.rk4_time_derivative(x) if args.use_rk4 else model.time_derivative(x)
        loss = L2_loss(dxdt, dxdt_hat)
        loss.backward() ; optim.step() ; optim.zero_grad()
        
        # run test data
        test_dxdt_hat = model.rk4_time_derivative(test_x) if args.use_rk4 else model.time_derivative(test_x)
        test_loss = L2_loss(test_dxdt, test_dxdt_hat)

        # logging
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        if args.verbose and step % args.print_every == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

    train_dxdt_hat = model.time_derivative(x)
    train_dist = (dxdt - train_dxdt_hat)**2
    test_dxdt_hat = model.time_derivative(test_x)
    test_dist = (test_dxdt - test_dxdt_hat)**2
    print('Final train loss {:.4e} +/- {:.4e}\nFinal test loss {:.4e} +/- {:.4e}'
        .format(train_dist.mean().item(), train_dist.std().item()/np.sqrt(train_dist.shape[0]),
                test_dist.mean().item(), test_dist.std().item()/np.sqrt(test_dist.shape[0])))

    return model, stats

if __name__ == "__main__":
    args = get_args()
    model, stats = train(args)

    # save
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
    label = '-baseline' if args.baseline else '-hnn'
    label = '-rk4' + label if args.use_rk4 else label
    label = label + '-rad' if args.rad else label
    path = '{}/{}{}.tar'.format(args.save_dir, args.name, label)
    torch.save(model.state_dict(), path)