import numpy as np
import sys
from m1 import neural_network
import activation_functions
import loss
import utilities
import regularizers

# main for M1. 

# verify number args
if len(sys.argv) < 5:
    print("Error: python3 main_1.py <algorithm_type> <alg_file> <output_file> <config_nn_file>")
    sys.exit(1)
# 1 or 2 
algorithm_type = sys.argv[1]
# algorithm configuration file
alg_file = sys.argv[2]
# output file
output_file = sys.argv[3]

# nn config file 
config_nn_file = sys.argv[4]
(training_set, topology, epochs, minibatch_size, reg_coef) = utilities.read_config_nn(config_nn_file)
# dataset
(inputs, targets) = utilities.read_csv(training_set)

# topology 
topology = np.insert(topology, 0, len(inputs[0])) 
if isinstance(targets[0], np.ndarray):
    topology = np.append(topology, len(targets[0]))
else:
    topology = np.append(topology, 1)

# regularization
reg = regularizers.L2(reg_coef)

# NN creation
act_hidden = activation_functions.relu
act_output = activation_functions.linear
loss_function = loss.least_squared
net = neural_network(topology, act_hidden, act_output, loss_function, reg)
net.set_algorithm(algorithm_type, alg_file)
# training
loss_tot = net.run_training(inputs, targets, epochs, minibatch_size)
# saving 
np.save(output_file, loss_tot)