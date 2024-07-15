import numpy as np
import activation_functions
from m1 import neural_network
from gradient import gradient
import utilities

# HB method

class A1:
    
    def __init__(self, nn : neural_network, filename):
        (k,beta,self.stepsize,self.momentum,) = utilities.read_config_alg(filename)
        self.nn = nn

    # backpropagation, updating weigths and bias 
    def learning(self, loss_deriv, hidden_outputs, minibatch_data, grad_old, epoch_loss ):
        grad_curr = self.backpropagation(loss_deriv, hidden_outputs, minibatch_data)
        for j in range(len(self.nn.weigths.hiddens)):
            self.nn.weigths.hiddens[j] = self.update_weights(self.nn.weigths.hiddens[j], grad_curr.hiddens[j], grad_old.hiddens[j])
            self.nn.weigths.hid_bias[j] = self.update_weights(self.nn.weigths.hid_bias[j], grad_curr.hid_bias[j], grad_old.hid_bias[j])
        self.nn.weigths.out = self.update_weights(self.nn.weigths.out, grad_curr.out , grad_old.out)
        self.nn.weigths.out_bias = self.update_weights(self.nn.weigths.out_bias, grad_curr.out_bias, grad_old.out_bias)
        return grad_curr


    # backpropagation
    # hidden_outputs[i] is the output of the i-th hidden layer for the considered pattern
    def backpropagation(self, d_loss, hidden_outputs , data):
        # creating an instance of the gradient class
        grad = gradient(self.nn.topology)
        # computing gradient
        input = hidden_outputs[-1]
        next_weigths = self.nn.weigths.out
        delta_out = activation_functions.derivative(self.nn.act_output)(np.dot(input, next_weigths)) * d_loss
        grad.out = np.outer(input, delta_out) 
        grad.out_bias = delta_out
        delta_hiddens = []
        
        d_f = activation_functions.derivative(self.nn.act_hidden) 
        
        
        # hidden layers gradients
        if len(self.nn.weigths.hiddens) == 1: # single hidden layer 
            next_weigths = np.transpose(self.nn.weigths.out)
            delta_hiddens.append(np.dot(delta_out, next_weigths)*d_f(np.dot(data,self.nn.weigths.hiddens[0])))
            grad.hiddens.append(np.outer(data, delta_hiddens[-1]))
            grad.hid_bias.append(delta_hiddens[-1])

        else:
            for i in range(len(self.nn.weigths.hiddens)-1, -1, -1):
                
                delta = delta_out if i == len(self.nn.weigths.hiddens)-1 else delta_hiddens[-1]
                input = data if i == 0 else hidden_outputs[i-1]
                next_weigths = np.transpose(self.nn.weigths.out) if i == len(self.nn.weigths.hiddens)-1 else np.transpose(self.nn.weigths.hiddens[i+1])
                
                delta_hiddens.append(np.dot(delta, next_weigths)*d_f(np.dot(input,self.nn.weigths.hiddens[i])))
                grad.hiddens.append(np.outer(input, delta_hiddens[-1]))
                grad.hid_bias.append(delta_hiddens[-1])
        
        # matrices order inversion
        grad.hiddens = grad.hiddens[::-1]      
        grad.hid_bias = grad.hid_bias[::-1]

        return grad
    

    # updating weigths, with momentum and regularization
    def update_weights(self, weigths, gradients, old_gradients):
        weigths = weigths - self.stepsize*gradients + self.momentum*old_gradients 
        if self.nn.regularizer:
            weigths = weigths - self.nn.regularizer.gradient(weigths)
        return weigths

