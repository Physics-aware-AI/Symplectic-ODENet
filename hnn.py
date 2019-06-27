# code borrowed from Sam Greydanus
# https://github.com/greydanus/hamiltonian-nn

import torch
import numpy as np

from nn_models import MLP


class HNN(torch.nn.Module):
    def __init__(self, input_dim, differentiale_model, device,
                    baseline=False, assume_canonical_coords=True,
                    damp=False, dampNet=None):
        super(HNN, self).__init__()
        self.baseline = baseline
        self.differentiale_model = differentiale_model
        self.assume_canonical_coords = assume_canonical_coords
        self.device = device
        self.M = self.permutation_tensor(input_dim)
        self.nfe = 0
        self.damp = damp

        if damp:
            self.dampNet = dampNet
        

    def forward(self, x):
        if self.baseline:
            return self.differentiale_model(x)
        else:
            y = self.differentiale_model(x)
            # assert y.dim() == 2 and y.shape[1] == 1
            return y

    # note that the input of this function has changed to meet the ODENet requirement.
    def time_derivative(self, t, x):
        self.nfe += 1
        if self.baseline:
            return self.differentiale_model(x)
        else:
            H = self.forward(x) # the Hamiltonian
            dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
            H_vector_field = torch.matmul(dH, self.M.t())
            if not self.damp:
                return H_vector_field
            else:
                D = self.dampNet(x) # (100, 2, 2)
                D_vector_field = torch.squeeze(
                    torch.matmul(
                        torch.unsqueeze(dH, 1),
                        D # should be transpose but symmetric
                    )
                )
                return H_vector_field - D_vector_field



    def permutation_tensor(self, n):
        M = None
        if self.assume_canonical_coords:
            M = torch.eye(n)
            M = torch.cat([M[n//2:], -M[:n//2]])
        else:
            '''Constructs the Levi-Civita permutation tensor'''
            M = torch.ones(n,n) # matrix of ones
            M *= 1 - torch.eye(n) # clear diagonals
            M[::2] *= -1 # pattern of signs
            M[:,::2] *= -1
    
            for i in range(n): # make asymmetric
                for j in range(i+1, n):
                    M[i,j] *= -1
        return M.to(self.device)


class HNN_structure(torch.nn.Module):
    def __init__(self, input_dim, L_net, V_net, device,
                    assume_canonical_coords=True):
        super(HNN_structure, self).__init__()
        self.L_net = L_net
        self.V_net = V_net
        self.device = device
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim)
        self.nfe = 0

    def forward(self, x):
        q, p = torch.chunk(x, 2, dim=1)
        L = self.L_net(q)
        V_q = self.V_net(q)
        M_q = L * L + 0.1

        H = p * p / M_q /2 + V_q
        return H
        # return p * p/ (2*self.L_net(q)*self.L_net(q)) + self.V_net(q)

    def time_derivative(self, t, x):
        self.nfe += 1
        H = self.forward(x)
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        vector_field = torch.matmul(dH, self.M.t())
        return vector_field


    def permutation_tensor(self, n):
        M = None
        if self.assume_canonical_coords:
            M = torch.eye(n)
            M = torch.cat([M[n//2:], -M[:n//2]])
        else:
            '''Constructs the Levi-Civita permutation tensor'''
            M = torch.ones(n,n) # matrix of ones
            M *= 1 - torch.eye(n) # clear diagonals
            M[::2] *= -1 # pattern of signs
            M[:,::2] *= -1
    
            for i in range(n): # make asymmetric
                for j in range(i+1, n):
                    M[i,j] *= -1
        return M.to(self.device)