import numpy as np

# dataset params
amp_range = [0.1, 5.0]
phase_range = [0, np.pi]
input_range = [-5.0, 5.0]

# model params
dim_input = 1
dim_output = 1
dim_hidden = [40,40]

# training params
task_lr = 1e-3
meta_lr = 1e-3
task_batch_size = 5   # number of samples, i.e. K in K-shot learning
meta_batch_size = 25  # number of tasks sampled per meta update in metalearning setting

# saving model
saving_freq = 100
