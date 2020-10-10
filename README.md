This is the code of Dynamic Learning Rate (DLR), a gradient descent algorithm for networks with back-propagation.
The documentary of the algorithm is currently on arXiv: https://arxiv.org/abs/2009.12745.
To implement DLR, you only need to change the constant learning rate used in the traditional stochastic gradient descent in back-propagation with something like lines 130 and 131 (search the function "learning_rate_scaling") in KleinFunction_DLR.py.
