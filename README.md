# CM_project_unipi

## To run the NN:  
python3 main_1.py <alg_type> <config_alg> <result_file> <config_nn>

alg_type: 1 for HB (a1), 2 for DS (a2). 
config_alg default: config/config_a1.txt for HB, config/config_a2.txt for DS
result_file: specify a file name to save the weights at the end of training (useful for testing)
config_nn default: config/config_nn.txt

## To run the LR solver:
python3 main_2.py
