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

    def Hamiltonian_vector(self, x):
        H = self.forward(x) # the Hamiltonian
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        H_vector_field = torch.matmul(dH, self.M.t())
        return H_vector_field


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

class HNN_structure_pend(torch.nn.Module):
    def __init__(self, input_dim, H_net=None, M_net=None, V_net=None, F_net=None, device=None,
                    assume_canonical_coords=True, baseline=False, structure=False):
        super(HNN_structure_pend, self).__init__()
        self.baseline = baseline
        self.structure = structure
        if self.structure:
            self.M_net = M_net
            self.V_net = V_net
            self.F_net = F_net
        else:
            self.H_net = H_net
            self.F_net = F_net

        self.device = device
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim)
        self.nfe = 0

    def forward(self, x):
        if self.baseline:
            return self.H_net(x)
        if self.structure:
            bs = x.shape[0]

            q, p, p_dot = torch.chunk(x, 3, dim=1)
            V_q = self.V_net(q)
            # M_q_inv = self.M_net(q)
            M_q_inv = torch.tensor([1.0, 1.0, 12.0], dtype=torch.float32, device=self.device)
            M_q_inv = torch.unsqueeze(torch.diag_embed(M_q_inv), dim=0)
            M_q_inv = M_q_inv.repeat(bs,1,1)

            p_aug = torch.unsqueeze(p, dim=2)
            H = torch.squeeze(torch.matmul(torch.transpose(p_aug, 1, 2), torch.matmul(M_q_inv, p_aug)))/2.0 + torch.squeeze(V_q)
        else:
            H = self.H_net(x)

        return H

    def time_derivative(self, t, x):
        self.nfe += 1
        if self.baseline:
            return self.H_net(x)
        if self.structure:
            bs = x.shape[0]

            q, p, p_dot = torch.chunk(x, 3, dim=1)
            V_q = self.V_net(q)
            # M_q_inv = self.M_net(q)
            M_q_inv = torch.tensor([1.0, 1.0, 12.0], dtype=torch.float32, device=self.device)
            M_q_inv = torch.unsqueeze(torch.diag_embed(M_q_inv), dim=0)
            M_q_inv = M_q_inv.repeat(bs,1,1)

            p_aug = torch.unsqueeze(p, dim=2)
            H = torch.squeeze(torch.matmul(torch.transpose(p_aug, 1, 2), torch.matmul(M_q_inv, p_aug)))/2.0 + torch.squeeze(V_q)
        else:
            H = self.H_net(x)
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        H_vector_field = torch.matmul(dH[:, 0:6], self.M.t())
        dHdq, dHdp, _ = torch.chunk(dH, 3, dim=1)

        # dHdq1 = torch.autograd.grad(V_q.sum(), q, create_graph=True)[0]
        # dHdp1 = torch.squeeze(torch.matmul(M_q_inv, p_aug), dim=2)

        F = self.F_net(x)
        num = torch.squeeze(torch.bmm(torch.unsqueeze(dHdp, 1), torch.unsqueeze(F, 2)), dim=2)
        num = num * dHdp
        den = torch.squeeze(torch.bmm(torch.unsqueeze(dHdp, 1), torch.unsqueeze(dHdp, 2)), dim=2)
        div = num/den
        div = torch.where(torch.isnan(div), torch.zeros_like(div), div)

        Fc = F - div

        Fc_vector_field = torch.cat((torch.zeros_like(Fc), Fc), dim=1)

        return torch.cat((H_vector_field + Fc_vector_field, torch.zeros_like(Fc)), dim=1)

        # A_q = self.A_net(q)
        # A_T_q = torch.transpose(A_q, 1, 2)

        # # calculate the matrix to be differentiate
        # A_T_M_inv_p = torch.matmul(A_T_q, torch.matmul(M_q_inv, torch.unsqueeze(p, dim=2)))
        # dA_T_M_inv_p_dq = self.get_jacobian(q, A_T_M_inv_p)

        # # calculate lambda
        # RHS = torch.matmul(A_T_q, torch.matmul(M_q_inv, torch.unsqueeze(dHdq, dim=2))) \
        #         - torch.matmul(dA_T_M_inv_p_dq, torch.unsqueeze(dHdp, dim=2))
        # LHS = torch.matmul(A_T_q, torch.matmul(M_q_inv, A_q))
        # LHS = LHS + torch.ones_like(LHS) * 0.1
        # lambd = torch.matmul(torch.inverse(LHS), RHS)

        # con_force = torch.squeeze(torch.matmul(A_q, lambd))
        # con_force_vector_field = torch.cat((torch.zeros_like(con_force), con_force), dim=1)

    # def get_jacobian(self, input, output):
    #     output = torch.squeeze(output)
    #     output_sum = output.sum(0)

    #     jac = []
    #     for i in range(len(output_sum)):
    #         jac.append(torch.autograd.grad(output_sum[i], input, create_graph=True)[0])
    #     return torch.stack(jac, dim=1)

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

    def get_intermediate_value(self, t, x):
        bs = x.shape[0]

        q, p = torch.chunk(x, 2, dim=1)
        V_q = self.V_net(q)
        # M_q_inv = self.M_net(q)
        M_q_inv = torch.tensor([1.0, 1.0, 12.0], dtype=torch.float32, device=self.device)
        M_q_inv = torch.unsqueeze(torch.diag_embed(M_q_inv), dim=0)
        M_q_inv = M_q_inv.repeat(bs,1,1)

        p_aug = torch.unsqueeze(p, dim=2)
        H = torch.squeeze(torch.matmul(torch.transpose(p_aug, 1, 2), torch.matmul(M_q_inv, p_aug)))/2.0 + torch.squeeze(V_q)

        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        H_vector_field = torch.matmul(dH, self.M.t())
        dHdq, dHdp = torch.chunk(dH, 2, dim=1)

        # dHdq1 = torch.autograd.grad(V_q.sum(), q, create_graph=True)[0]
        # dHdp1 = torch.squeeze(torch.matmul(M_q_inv, p_aug), dim=2)

        F = self.F_net(torch.cat((q, p, dHdq), dim=1))
        num = torch.squeeze(torch.bmm(torch.unsqueeze(dHdp, 1), torch.unsqueeze(F, 2)), dim=2)
        num = num * dHdp
        den = torch.squeeze(torch.bmm(torch.unsqueeze(dHdp, 1), torch.unsqueeze(dHdp, 2)), dim=2)
        Fc = F - num/den

        Fc_vector_field = torch.cat((torch.zeros_like(Fc), Fc), dim=1)

        dHdp_Fc = torch.squeeze(torch.bmm(torch.unsqueeze(dHdp, 1), torch.unsqueeze(Fc, 2)), dim=2)
        return dHdq, dHdp, F, Fc, dHdp_Fc

class HNN_structure_forcing(torch.nn.Module):
    def __init__(self, input_dim, H_net=None, M_net=None, V_net=None, g_net=None, device=None,
                    assume_canonical_coords=True, baseline=False, structure=False):
        super(HNN_structure_forcing, self).__init__()
        self.baseline = baseline
        self.structure = structure
        if self.structure:
            self.M_net = M_net
            self.V_net = V_net
            self.g_net = g_net
        else:
            self.H_net = H_net
            self.g_net = g_net

        self.device = device
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim)
        self.nfe = 0

    def forward(self, x):
        if self.baseline:
            return self.H_net(x)
        if self.structure:
            bs = x.shape[0]

            q, p, u = torch.chunk(x, 3, dim=1)
            V_q = self.V_net(q)
            M_q_inv = self.M_net(q)
            # M_q_inv = torch.tensor([1.0, 1.0, 12.0], dtype=torch.float32, device=self.device)
            # M_q_inv = torch.unsqueeze(torch.diag_embed(M_q_inv), dim=0)
            # M_q_inv = M_q_inv.repeat(bs,1,1)

            p_aug = torch.unsqueeze(p, dim=2)
            H = torch.squeeze(torch.matmul(torch.transpose(p_aug, 1, 2), torch.matmul(M_q_inv, p_aug)))/2.0 + torch.squeeze(V_q)
        else:
            H = self.H_net(x)

        return H

    def time_derivative(self, t, x):
        self.nfe += 1
        if self.baseline:
            dq, dp , _ =  torch.chunk(self.H_net(x), 3, dim=1)
            return torch.cat((dq, dp, torch.zeros_like(dq)), dim=1)
        if self.structure:
            bs = x.shape[0]

            q, p, u = torch.chunk(x, 3, dim=1)
            V_q = self.V_net(q)
            M_q_inv = self.M_net(q)
            # M_q_inv = torch.tensor([1.0, 1.0, 12.0], dtype=torch.float32, device=self.device)
            # M_q_inv = torch.unsqueeze(torch.diag_embed(M_q_inv), dim=0)
            # M_q_inv = M_q_inv.repeat(bs,1,1)

            p_aug = torch.unsqueeze(p, dim=2)
            H = torch.squeeze(torch.matmul(torch.transpose(p_aug, 1, 2), torch.matmul(M_q_inv, p_aug)))/2.0 + torch.squeeze(V_q)
        else:
            q, p, u = torch.chunk(x, 3, dim=1)
            H = self.H_net(x)
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        # H_vector_field = torch.matmul(dH, self.M.t())
        dHdq, dHdp, _ = torch.chunk(dH, 3, dim=1)
        H_vector_field = torch.cat((dHdp, -dHdq, torch.zeros_like(dHdq)), dim=1)

        g_q = self.g_net(q)

        F = g_q * u

        Fc_vector_field = torch.cat((torch.zeros_like(F), F, torch.zeros_like(F)), dim=1)

        return H_vector_field + F_vector_field

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