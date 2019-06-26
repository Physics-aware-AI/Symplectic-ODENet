# Learning dyanmics using ODENet and HNN

Using ODENet would allow us to learn the dynamcis without using the "accelaration data". First we want to investigate what the results would be when HNN and ODENet are directly combined together. one technical detail along this line is that how to make a fair comparison. With the ODENet architecture, we can predict a series of data points forward in time given an initial condition. The most ideal comparison would be letting ODENet predicting one point ahead and comparing the predicted values and the true values. But the number of predicted points could be viewed as a parameter and we could investigate the quality of the results when this parameter changes. 

Todo: 
1. generate the results for different numbers of predicted points. Analyze the different performance in this context
2. try to enforce the **mass matrix** structure into the Hamiltonian.
3. try to enforce the **damping** structure into the Hamiltonian.
4. after these has been done. Implement the model to investigate the real pendulum data. 

## Hamiltonian dynamics of a pendulum 
$H(q, p) = \frac{l^2p^2}{2m} + 2mgl(1-cosq)$

$\dot{q} = \frac{\partial H}{\partial p}$

$\dot{p} = -\frac{\partial H}{\partial q}$

### The original result in the HNN paper, training set $(q, p, \dot{q}, \dot{p})$

![image](figures/pend.png)
![image](figures/pend-integration.png)

### Adding ODENet to HNN, training set $(q,p)$, evaluate 2 points

![image](figures/pend-p2.png)
![image](figures/pend-p2-integration.png)

### Adding ODENet to HNN, training set $(q,p)$, evaluate 3 points

![image](figures/pend-p3.png)
![image](figures/pend-p3-integration.png)

### Adding ODENet to HNN, training set $(q,p)$, evaluate 4 points

![image](figures/pend-p4.png)
![image](figures/pend-p4-integration.png)

### Adding ODENet to HNN, training set $(q,p)$, evaluate 5 points

![image](figures/pend-p5.png)
![image](figures/pend-p5-integration.png)

## Try to add structure to the ODENet

It takes some trial and error to figure out how to implement the mass matrix structure into the network, even in the 1D case
- First I parametrize the mass matrix as $M = L * L$. Since mass shows up in the denominator. in other words, mass inverse shows up. This scheme would cause issue in the traning procedure. My guess is that it's numerically unstable and some weights blow up
- Next I try to add a constant to the mass $ M = L * L + 0.1$. I think the positive constant stablized the process and the results are satisfatory.
- Biswa suggested learning the inverse of mass directly. so I tried $H = p * p * M_q /2 + V_q$ instead of $H = p * p / M_q /2 + V_q$, with $M = L * L$. Training looks correct but we actually learnt the wrong vector field.