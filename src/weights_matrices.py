import numpy as np

class weigths_matrices: 
    def __init__(self,topology):
        self.hiddens = []
        self.hid_bias = []
        np.random.seed(47)
        # per gli hidden  
        for i in range(len(topology)-2):
            self.hiddens.append(np.random.uniform(-0.5,0.5, (topology[i] , topology[i+1]) ))
            # self.hid_bias.append(np.full(topology[i+1], 0))
            self.hid_bias.append(np.random.uniform(-0.5,0.5,  topology[i+1] ))
        # per l'output layer 
        self.out = np.random.uniform(-0.5,0.5, (topology[-2], topology[-1]) )
        self.out_bias = np.random.uniform(-0.5,0.5,  topology[-1])


    # (used in tomography)
    def random_weigths(self, topology):
        self.hiddens = []
        self.hid_bias = []
        np.random.seed(47)
        # hidden layers
        for i in range(len(topology)-2):
            self.hiddens.append(np.random.uniform(-1, 1,(topology[i] , topology[i+1]) ))
            self.hid_bias.append(np.random.uniform(-1, 1, topology[i+1] ))
        # output layer 
        self.out = np.random.uniform(-1, 1,(topology[-2], topology[-1]) )
        self.out_bias = np.random.uniform(-1, 1, topology[-1])


    def get_weights_as_vector(self):
        w = []
        w.extend([item for row in self.out for item in row])
        w.extend(np.ravel(self.out_bias))
        for i in range(len(self.hiddens)):
            w.extend([item for row in self.hiddens[i] for item in row])
            w.extend(np.ravel(self.hid_bias[i]))
        return np.array(w)
    
    def set_weigths_from_vector(self, w):
        idx = 0
        
        # set out weights 
        out_shape = self.out.shape
        out_size = out_shape[0] * out_shape[1]
        self.out = w[idx:idx+out_size].reshape(out_shape)
        idx += out_size
        
        # set out bias 
        out_bias_size = self.out_bias.shape[0]
        self.out_bias = w[idx:idx+out_bias_size]
        idx += out_bias_size
        
        # set hidden weights and bias  
        for i in range(len(self.hiddens)):
            hid_shape = self.hiddens[i].shape
            hid_size = hid_shape[0] * hid_shape[1]
            self.hiddens[i] = w[idx:idx+hid_size].reshape(hid_shape)
            idx += hid_size
            
            hid_bias_size = self.hid_bias[i].shape[0]
            self.hid_bias[i] = w[idx:idx+hid_bias_size]
            idx += hid_bias_size
        
