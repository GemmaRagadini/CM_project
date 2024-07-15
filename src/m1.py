import loss
import numpy as np
import utilities
from weights_matrices import weigths_matrices
from gradient import gradient
import sys

# minumum found value 
opt_loss = 0.00036446917854614
           
class neural_network:

    def __init__(self, topology, activationFunctionForHidden, activationFunctionForOutput, lossFunction, regularizer):
        
        self.act_hidden = activationFunctionForHidden
        self.act_output = activationFunctionForOutput
        self.lossFunction = lossFunction
        # units list for each layer 
        # ex. [10,100,3]
        self.topology = topology 
        # regularization type
        self.regularizer = regularizer
        # weights matrices 
        self.weigths = weigths_matrices(self.topology) 
        self.min_loss = sys.maxsize      


    def set_algorithm(self, algorithm_type, file_alg):
        if int(algorithm_type) == 1:
            from a1 import A1
            self.algorithm = A1(self,file_alg)
        if int(algorithm_type) == 2:
            from a2 import A2  
            self.algorithm = A2(self,file_alg)  

    # forward propagation for single layer
    def forwardpropagation(self, data):
        hidden_outputs = []
        # hidden layers
        for i in range(len(self.weigths.hiddens)):
            if i == 0:
                hidden_outputs.append(np.ravel(self.act_hidden(np.dot(data, self.weigths.hiddens[i]) + self.weigths.hid_bias[i])))
            else: 
                hidden_outputs.append(np.ravel(self.act_hidden(np.dot(hidden_outputs[i-1], self.weigths.hiddens[i]) + self.weigths.hid_bias[i])))
        final_output = np.ravel(self.act_output(np.dot(hidden_outputs[-1], self.weigths.out) + self.weigths.out_bias))
        # final output: correspondent to output layer 
        # hidden_outputs: output array for each hidden layer  
        return (hidden_outputs,final_output) 
    

    # forward propagatin for minibatch
    def compute_minibatch(self, minibatch_data):
        hidden_outputs = []
        final_outputs = []
        for input_data in minibatch_data:
            outputs = self.forwardpropagation(input_data)
            hidden_outputs.append(outputs[0])
            final_outputs.append(outputs[1])
        return (utilities.avg_for_layer(hidden_outputs), np.divide(final_outputs,len(minibatch_data)) )



    def run_training(self, tr_data, tr_targets, numberEpochs, minibatch_size):
        loss_tot = []
       
        grad = gradient(self.topology)
        # training
        for epoch in range(numberEpochs):
            print("Epoca: ", epoch)
            epoch_loss_sum = 0
            for i in range(int(len(tr_data)/minibatch_size)):

                minibatch_data  = tr_data[i*minibatch_size:i*minibatch_size + minibatch_size]
                minibatch_target = tr_targets[i*minibatch_size:i*minibatch_size + minibatch_size]
                (avg_hidden_outputs, avg_final_output) = self.compute_minibatch(minibatch_data)
                avg_loss = self.lossFunction(np.divide(minibatch_target,len(minibatch_target)),avg_final_output)
                if self.regularizer:
                    avg_loss += self.regularizer(self.weigths.out) 
                    for j in range(len(self.weigths.hiddens)):
                        avg_loss += self.regularizer(self.weigths.hiddens[j])
                    

                d_loss = loss.derivative(self.lossFunction)(np.divide(minibatch_target, len(minibatch_target)),avg_final_output)
                if len(loss_tot) == 0:
                    epoch_loss = sys.maxsize
                else:
                    epoch_loss = loss_tot[-1]
              
                grad = self.algorithm.learning(d_loss, avg_hidden_outputs,  np.average(minibatch_data,axis=0) , grad , epoch_loss)
             
                epoch_loss_sum += avg_loss 

            for i in range(len(tr_data)%minibatch_size):
                if i == 0:
                    minibatch_size+=1 

                minibatch_data  = tr_data[-i:]
                minibatch_target = tr_targets[-i:]
                (avg_hidden_outputs, avg_final_output) = self.compute_minibatch(minibatch_data)
                avg_loss = self.lossFunction(np.divide(minibatch_target,len(minibatch_target)),avg_final_output)            
                if self.regularizer:
                    avg_loss += self.regularizer(self.weigths.out) 
                    for j in range(len(self.weigths.hiddens)):
                        avg_loss += self.regularizer(self.weigths.hiddens[j])

               
                d_loss = loss.derivative(self.lossFunction)(np.divide(minibatch_target, len(minibatch_target)),avg_final_output)
                print(d_loss)
                epoch_loss = loss_tot[-1]
                grad = self.algorithm.learning(d_loss, avg_hidden_outputs,np.average(minibatch_data,axis=0),grad , epoch_loss)
                epoch_loss_sum += avg_loss 

            # avg loss for epoch 
            loss_tot.append(epoch_loss_sum/len(tr_data))
            # last loss    
            print(loss_tot[-1])
            if loss_tot[-1] < self.min_loss:
                self.min_loss = loss_tot[-1]

        # saving
        np.savetxt("loss_per_epochs.txt", np.reshape(loss_tot,(-1, 1)), fmt='%.14f')
        print(self.min_loss)
        utilities.plot_error(np.log(np.divide(np.subtract(loss_tot,opt_loss), opt_loss)))
        
        return loss_tot

    #   for tomography 
    def compute_loss(self, x,y):
        loss_values = []
        for i in range(len(x)):
            out = self.forwardpropagation(x[i])[1]    
            loss_values.append(self.lossFunction(out,y[i]))
 
        return np.average(loss_values)
  