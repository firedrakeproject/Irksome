import matplotlib.pyplot as plt
import numpy

A = numpy.loadtxt("new_data.csv", delimiter=",")
t = A[:, 0]
CD = A[:, 1]
CL = A[:, 3]

plt.plot(t, CL)
plt.show()


