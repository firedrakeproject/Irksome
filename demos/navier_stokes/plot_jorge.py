import matplotlib.pyplot as plt
import numpy

A = numpy.loadtxt("new_data.csv", delimiter=",")
B = numpy.loadtxt("bdforces_lv3.csv", delimiter=",")
t = A[:, 0]
CD = A[:, 1]
CL = -A[:, 3]

tR = B[:, 1]
CDR = B[:, 3]
CLR = B[:, 4]

plt.plot(t, CL)
plt.plot(tR, CLR)
plt.show()

plt.plot(t, CD)
plt.plot(tR, CDR)
plt.show()
