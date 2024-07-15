import numpy as np 
import sys 
import time
import matplotlib.pyplot as plt
import utilities

# model 2 with algorithm 3
class m2:

    def __init__(self):
        pass

    def run(self, A, targets):
        # m samples number (input, target)
        # n entries numbers for each input
        # k entries numbers for each target 
        # m >> n => thin qr
        m = len(A)
        n = len(A[0])
        A = np.array(A)
        if isinstance(targets[0], np.ndarray): 
            B = np.array(targets)
        else:
            B = np.zeros((m,1)) # k = 1
            for i in range(m):
                B[i][0] = targets[i]
 
        # set algorithm
        algorithm = A3(A,B)
        self.X = algorithm.compute_x()


    def test(self, inputs, targets):
        m = len(inputs)
        A = np.array(inputs)
        if np.shape(A)[1] != np.shape(self.X)[0]:
            print("Errore: non corrispondenza dati train e test")
            sys.exit()

        # arccos(theta)
        cos_theta = np.linalg.norm(np.dot(A, self.X))/np.linalg.norm(targets)
        theta = np.arccos(cos_theta)

        # A condition number  
        k = utilities.cond_number(A)
        print("Condition number of A:", k)

        k_wrt_y = k / cos_theta
        print("Cond number of x wrt y <= ", k_wrt_y)

        # residual comparison for A3
        output1 = np.dot(A, self.X)
        print("Residuals:", np.linalg.norm(output1 - targets)/np.linalg.norm(targets))

        # residual comparison for np
        x, residuals, rank, s = np.linalg.lstsq(A, targets, rcond=None)
        output2 = np.dot(A, x)
        print("Residuals:", np.linalg.norm(output2 - targets)/np.linalg.norm(targets))
    


    # compute mean of 100 repetitions 
    # m : rows number 
    def measure_time(self, m, n=10, repeats=100, remove_extremes=True):
        np.random.seed(47)
        A = np.random.rand(m, n)
        times = []

        alg = A3(A, np.random.rand(m, 1))
        for _ in range(repeats):
            start_time = time.time()
            alg.thin_QR()
            end_time = time.time()
            times.append(end_time - start_time)

        if remove_extremes:
            lower_bound = np.percentile(times, 5)
            upper_bound = np.percentile(times, 95)
            times = [t for t in times if lower_bound <= t <= upper_bound]

        mean_time = np.mean(times)

        return mean_time
  
class A3:

    def __init__(self, A, B):
        self.A = A
        self.B = B
        pass

    def compute_x(self):
        (Q0,R0) = self.thin_QR()
     
        # A - QR difference 
        qr = np.linalg.qr(self.A,mode='reduced')
        print("Relative diff A and QR for np:", np.linalg.norm(self.A - np.dot(qr[0],qr[1]))/np.linalg.norm(self.A))
        print("Relative diff A and QR for A3:", np.linalg.norm(self.A - np.dot(Q0,R0))/np.linalg.norm(self.A))
      
        return np.linalg.solve(R0,np.dot(np.transpose(Q0),self.B))

    
    def householder_reflection(self, a):
        v = a.copy()
        s = np.linalg.norm(a)
        v[0] += np.sign(a[0]) * s
        v /= np.linalg.norm(v)
        return v


    def thin_QR(self):
        m, n = np.shape(self.A)
        R = self.A.copy()
        u_vectors = []

        for i in range(n):
            v = self.householder_reflection(R[i:, i])
            u_vectors.append(v)
            R[i:, i:] -= 2 * np.outer(v, (np.dot(np.transpose(v), R[i:, i:])))

        # compute Q0
        Q_0 = np.eye(m, n)
        for i in reversed(range(n)):  
            v = u_vectors[i]
            Q_0[i:, :] -= 2 * np.outer(v, np.dot(v, Q_0[i:, :]))

        return Q_0, R[:n, :]
