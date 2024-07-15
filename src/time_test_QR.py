import linear_regression
import utilities
import matplotlib.pyplot as plt 
import numpy as np 

solver = linear_regression.m2()

m_values = range(10, 800, 10)
times = []

# Measure the execution times of the decomposition for various dimensions of m.
for m in m_values:
    elapsed_time = solver.measure_time(m)
    times.append(elapsed_time)
    print(f"m={m}, Time={elapsed_time:.4f} seconds")

# plot execution times
plt.plot(m_values, times, marker='o')
plt.xlabel('Number of rows', fontsize=14)
plt.ylabel('Execution time', fontsize=14)
plt.grid(True)
plt.show()

