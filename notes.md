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

### Adding ODENet to HNN, training set $(q,p)$, evaluate 3 poitns

![image](figures/pend-p3.png)
![image](figures/pend-p3-integration.png)

### Adding ODENet to HNN, training set $(q,p)$, evaluate 4 poitns

![image](figures/pend-p4.png)
![image](figures/pend-p4-integration.png)

### Adding ODENet to HNN, training set $(q,p)$, evaluate 3 poitns

![image](figures/pend-p5.png)
![image](figures/pend-p5-integration.png)