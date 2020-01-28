# Symplectic ODE-Net | 2019
# Yaofeng Desmond Zhong, Biswadip Dey, Amit Chakraborty

# Generate figure of training error and prediction error.
# The statistics in this script are obtained by training models for 4 tasks
# under the configuration num_points=4 and solver='rk4'.

# Number of training trajectories are varied from 16 to 1024.
# All the trained models are stored in the previous commits in this repo
# using git-lfs

# All the statistics can be verified by going back to those commits and 
# running the 'analyze' script or jupyter notebook of each task.

#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
DPI = 600
FORMAT = 'pdf'
LINE_SEGMENTS = 10
ARROW_SCALE = 40
ARROW_WIDTH = 6e-3
LINE_WIDTH = 2

def get_args():
    return {'fig_dir': './figures'}

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

args = ObjectView(get_args())


#%%
x_axis = np.array([16, 32, 64, 128, 256, 512, 1024])

force_base = np.array([39.4, 39.14, 30.815, 35.356, 33.986, 33.789, 33.986])
force_symoden = np.array([0.58254, 0.67568, 0.89116, 0.83255, 0.89219, 1.8251, 0.89219])
force_symoden_struct = np.array([0.77013, 0.53097, 1.5019, 0.75556, 0.64103, 0.661, 0.64103])
pred_force_base = np.array([95, 28.179, 37.875, 50.679, 32.059, 31.677, 33.979])
pred_force_symoden = np.array([23.061, 21.809, 17.169, 17.842, 17.088, 25.48, 24.057])
pred_force_symoden_struct = np.array([8.3195, 5.8331, 23.948, 21.003, 14.783, 18.128, 22.164])

embed_naive = np.array([2.2598, 1.9373, 2.3095, 3.3458, 1.9881, 6.8656, 7.8169])
embed_base = np.array([0.51217, 0.57085, 0.5864, 0.52623, 0.53352, 0.53107, 0.53793])
embed_symoden = np.array([1.6384, 2.0384, 1.7607, 1.6299, 1.5606, 1.6090, 1.8241])
embed_symoden_struct = np.array([0.061244, 0.064135, 0.067448, 0.072003, 0.074001, 0.076196, 0.075709])

pred_embed_naive = np.array([272.74, 208.94, 317.21, 270.45, 283.44, 447.79, 426.79])
pred_embed_base = np.array([8.8225, 17.562, 14.305, 26.778, 17.476, 16.549, 2.9406])
pred_embed_symoden = np.array([7.2904, 5.1450, 3.69, 3.0387, 4.2426, 8.6148, 9.5405])
pred_embed_symoden_struct = np.array([0.23183, 0.23148, 0.1992, 0.18396, 0.26408, 0.40319, 0.94716])

cart_naive = np.array([10.601, 14.695, 15.533, 21.068, 22.12, 21.173, 21.401])
cart_base = np.array([0.36734, 0.43090, 0.44873, 0.56727, 0.6832, 0.832, 0.71988])
cart_symoden = np.array([5.9175, 24.562, 4.8388, 29.713, 29.318, 30.168, 28.125])
cart_symoden_struct = np.array([1.0047, 11.730, 1.7836, 2.1109, 2.1411, 1.9989, 1.9989])

pred_cart_naive = np.array([272.1, 251.94, 332.44, 253.43, 258.70, 243.98, 250.62])
pred_cart_base = np.array([138.7, 37.183, 52.262, 16.914, 21.990, 20.907, 20.674])
pred_cart_symoden = np.array([189.61, 636.39, 225.22, 795.17, 814.71, 854.19, 814.04])
pred_cart_symoden_struct = np.array([33.252, 77.723, 11.413, 12.085, 12.629, 11.732, 11.088])

double_naive = np.array([1.0345, 1.6478, 2.0402, 2.6397, 2.9444, 3.3754, 3.3966])
double_base = np.array([1.1619, 2.3207, 2.0666, 8.2426, 7.9885, 6.7069, 7.6092])
double_symoden = np.array([0.91524, 2.6558, 1.319, 1.2319, 1.5914, 1.3029, 1.4492])
double_symoden_struct = np.array([0.25096, 0.29942, 0.24631, 0.26576, 0.093754, 0.20858, 0.097612])

pred_double_naive = np.array([78.73, 72.054, 64.612, 66.809, 63.058, 77.376, 73.787])
pred_double_base = np.array([17.883, 37.273, 26.684, 149.64, 140.89, 116.44, 123.33])
pred_double_symoden = np.array([13.197, 19.228, 9.7211, 16.437, 21.559, 14.602, 15.47])
pred_double_symoden_struct = np.array([1.5161, 2.8168, 2.0734, 2.7492, 1.2519, 2.3067, 1.5636])


#%%
fig = plt.figure(figsize=(12, 4.8), dpi=DPI)
plt.subplot(2, 4, 1)
plt.plot(x_axis, force_base, 'ys-')
plt.plot(x_axis, force_symoden, 'gs-')
plt.plot(x_axis, force_symoden_struct, 'bs-')
plt.xscale('log')
plt.yscale('log')
# plt.xlabel('number of state initial condition')
plt.ylabel('Train error')
plt.title('Task 1: Pendulum')
# plt.legend(fontsize=6)

plt.subplot(2, 4, 5)
plt.plot(x_axis, pred_force_base, 'ys-')
plt.plot(x_axis, pred_force_symoden, 'gs-')
plt.plot(x_axis, pred_force_symoden_struct, 'bs-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('number of initial state conditions')
plt.ylabel('Prediction error')
# plt.legend(fontsize=6)

plt.subplot(2, 4, 2)
plt.plot(x_axis, embed_naive, 'ys-')
plt.plot(x_axis, embed_base, 'rs-')
plt.plot(x_axis, embed_symoden, 'gs-')
plt.plot(x_axis, embed_symoden_struct, 'bs-')
plt.xscale('log')
plt.yscale('log')
# plt.xlabel('number of training trajectories')
# plt.ylabel('Train loss')
plt.title('Task 2: Pendulum(embed)')
# plt.legend(fontsize=6)

plt.subplot(2, 4, 6)
plt.plot(x_axis, pred_embed_naive, 'ys-')
plt.plot(x_axis, pred_embed_base, 'rs-')
plt.plot(x_axis, pred_embed_symoden, 'gs-')
plt.plot(x_axis, pred_embed_symoden_struct, 'bs-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('number of initial state conditions')
# plt.ylabel('Prediction loss')
# plt.legend(fontsize=6)

plt.subplot(2, 4, 3)
plt.plot(x_axis, cart_naive, 'ys-')
plt.plot(x_axis, cart_base, 'rs-')
plt.plot(x_axis, cart_symoden, 'gs-')
plt.plot(x_axis, cart_symoden_struct, 'bs-')
plt.xscale('log')
plt.yscale('log')
# plt.xlabel('number of training trajectories')
# plt.ylabel('Train loss')
plt.title('Task 3: CartPole')
# plt.legend(fontsize=6)

plt.subplot(2, 4, 7)
plt.plot(x_axis, pred_cart_naive, 'ys-')
plt.plot(x_axis, pred_cart_base, 'rs-')
plt.plot(x_axis, pred_cart_symoden, 'gs-')
plt.plot(x_axis, pred_cart_symoden_struct, 'bs-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('number of initial state conditions')
# plt.ylabel('Prediction loss')
# plt.legend(fontsize=6)

plt.subplot(2, 4, 4)
plt.plot(x_axis, double_naive, 'ys-', label='Naive Baseline')
plt.plot(x_axis, double_base, 'rs-', label='Geometric Baseline')
plt.plot(x_axis, double_symoden, 'gs-', label='Unstructured SymODEN')
plt.plot(x_axis, double_symoden_struct, 'bs-', label='SymODEN')
plt.xscale('log')
plt.yscale('log')
plt.ylim([0.05, 200])
# plt.xlabel('number of training trajectories')
# plt.ylabel('Train loss')
plt.title('Task 4: Acrobot')
plt.legend(fontsize=6)

plt.subplot(2, 4, 8)
plt.plot(x_axis, pred_double_naive, 'ys-', label='Naive Baseline')
plt.plot(x_axis, pred_double_base, 'rs-', label='Geometric Baseline')
plt.plot(x_axis, pred_double_symoden, 'gs-', label='Unstructured SymODEN')
plt.plot(x_axis, pred_double_symoden_struct, 'bs-', label='SymODEN')
plt.xscale('log')
plt.yscale('log')
plt.ylim([1, 5000])
plt.xlabel('number of initial state conditions')
# plt.ylabel('Prediction loss')
plt.legend(fontsize=6)

plt.tight_layout()
# fig.savefig('{}/fig-train-pred-loss.{}'.format(args.fig_dir, FORMAT))

#%%
