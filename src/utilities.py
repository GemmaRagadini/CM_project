import numpy as np 
import matplotlib.pyplot as plt
import re
import csv

def read_config_nn(filename):
    
    tr = None
    topology = None
    epochs = 0
    minibatch_size = 0
    reg_coefficient = 0

    with open(filename, "r") as file:
        for line in file:
            match = re.match(r'^(\w+)\s*=\s*(.*)$', line)
            if match:
                key = match.group(1)
                value = match.group(2).strip()
                # Assegna i valori alle variabili in base alla chiave
                if key == "TR":
                    tr = value
                elif key == "topology":
                    topology = [int(x) for x in value.split(",")]
                elif key == "epochs":
                    epochs = int(value)
                elif key == "minibatch_size":
                    minibatch_size = int(value)
                elif key == "reg_coefficient":
                    reg_coefficient = float(value)
                

    return tr, topology, epochs, minibatch_size, reg_coefficient


def read_config_alg(filename):      
    stepsize = 0
    momentum = 0
    k = 0
    beta = 0
    with open(filename, "r") as file:
        for line in file:
            match = re.match(r'^(\w+)\s*=\s*(.*)$', line)
            if match:
                key = match.group(1)
                value = match.group(2).strip()
                if key == "k":
                    k = float(value)
                elif key == "beta":
                    beta = float(value)
                elif key == "stepsize":
                    stepsize = float(value)
                elif key == "momentum":
                    momentum = float(value)
    return k, beta, stepsize, momentum


# for ML-CUP
def read_csv(filename):
    inputs = []
    targets = []
    numero_di_riga_iniziale = 7 
    with open(filename, 'r') as file:
        lettore_csv = csv.reader(file)
        tutte_le_righe = list(lettore_csv)
        # Leggi solo le righe dalla posizione desiderata in avanti
        righe_selezionate = tutte_le_righe[numero_di_riga_iniziale:]
        righe_selezionate = np.array(righe_selezionate)
        righe_selezionate = righe_selezionate.astype(float)
        # divisione dei dati in input , output e il primo valore ignorato
        for riga in righe_selezionate:
            inputs.append(riga[1:11])
            targets.append(riga[-3:])
    return (inputs,targets)


# average per-layer outputs of the minibatch
def avg_for_layer(minibatch_outputs):  
    avg = []
    for i in range(len(minibatch_outputs[0])):
        minibatch_outputs[0][i] = np.ravel(minibatch_outputs[0][i])
        avg.append(np.zeros_like(minibatch_outputs[0][i]))
        for j in range(len(minibatch_outputs)):
            avg[-1] = np.add(avg[-1],minibatch_outputs[j][i])
        avg[-1]= np.divide(avg[-1],len(minibatch_outputs))

    return avg

# plot relative error 
def plot_error(x, lambda_value=None, filename='plot.png'):

    plt.figure(figsize=(10, 6))
    plt.plot(x, linewidth=3, color='b')
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("log(Er)", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    plt.show()


def is_upper_triangular(matrix):
    return np.allclose(matrix, np.triu(matrix))

def is_orthogonal(matrix):
    return np.allclose(np.dot(matrix, np.transpose(matrix)), np.eye(len(matrix)))


# compute A condition number from singular values 
def cond_number(A):
    U, S, Vt = np.linalg.svd(A)
    sigma_max = np.max(S)
    sigma_min = np.min(S[S > 0]) 
    kappa = sigma_max / sigma_min
    return kappa