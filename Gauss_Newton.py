import numpy as np
import math
import matplotlib.pyplot as plt

target = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]
youtcome = [0.9524, 1.6948, 1.9874, 1.8075, 1.1885, 0.2765, -0.7072, -1.5118, -1.9570, -1.9120,
     -1.4196, -0.5508, 0.4151, 1.3227, 1.8736, 1.9804, 1.5773, 0.8115, -0.1386, -1.0678, -1.7621]
iterations = 100
rows = 21
cols = 3

B = np.array([1.0, 1.0, 0.0])
B = B.reshape(3, -1)


Dr = np.zeros((rows, cols)) # Jacobian matrix from r
r = np.zeros((rows, 1)) #r equations


def partialDerB1(B2, B3, t):
    return math.sin(B2 * t + B3)


def partialDerB2(B1, B2, B3, t):
    return B1 * t * math.cos(B2 * t + B3)


def partialDerB3(B1,B2,B3,t):
    return B1 * math.cos(B2 * t + B3)


def rx(B1,B2,B3,t,y):
    return B1 * np.sin(B2 * t + B3) - y


def f(B1, B2, B3, t):
    return B1 * np.sin(B2 * t + B3)


for b in range(iterations):
    for j in range(rows):
        r[j, 0] = rx(B[0], B[1], B[2], target[j], youtcome[j])
        Dr[j, 0] = partialDerB1(B[1], B[2], target[j])
        Dr[j, 1] = partialDerB2(B[0], B[1], B[2], target[j])
        Dr[j, 2] = partialDerB2(B[0], B[1], B[2], target[j])
    Drt = Dr.T
    B = B - np.matmul(np.linalg.pinv(np.matmul(Drt, Dr)), np.matmul(Drt, r))

print(B)


plt.scatter(target, youtcome)

newt = np.linspace(0, 10)

plt.plot(newt, f(B[0], B[1], B[2], newt), 'b')
plt.show()