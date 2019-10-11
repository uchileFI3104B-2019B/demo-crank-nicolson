"""
Este script resuelve un problema simple de diffusion en 1D.
La ecuación a resover es:
    dT/dt = d2T/dx2;
    T(0,x) = sin(pi * x);
    T(t, 0) = T(t, 1) = 0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.linalg import solve

# Setup
Nx = 35
h = 1 / (Nx - 1)

N_temporales = 400
# eps = h**2 / 2
eps = 0.001

s = eps / 2 / h**2

T = np.zeros((N_temporales, Nx))

# Condicion Inicial
T[0] = np.sin(np.pi * np.arange(Nx)*h)

# matrices tridiagonales
M_derecha = diags((s, (1-2*s), s), (-1, 0, 1), shape=(Nx-2, Nx-2))
M_izquierda = diags((-s, (1 + 2*s), -s), (-1, 0, 1), shape=(Nx-2, Nx-2))


# Avances en el tiempo
for i in range(1, N_temporales):
    b = np.matmul(M_derecha.toarray(), T[i-1, 1:-1])
    T[i, 1:-1] = solve(M_izquierda.toarray(), b)

# Graficos
x = np.arange(Nx) * h
time = np.arange(N_temporales) * eps

plt.figure(1)
plt.clf()

for i in range(0, N_temporales, 25):
    label = "T={:.3f}".format(eps * i)
    plt.plot(x, T[i], label=label)

plt.xlabel('x')
plt.ylabel('Temperatura')
plt.legend()
plt.show()

plt.figure(2)
plt.clf()

X, Time = np.meshgrid(x, time)
plt.pcolormesh(X, Time, T)

plt.xlabel("x")
plt.ylabel("tiempo [unidades]")

plt.show()

plt.figure(3)
plt.clf()

plt.plot(time, T[:, int(0.5*(Nx-1))])
plt.xlabel("tiempo")
plt.ylabel('Temperatura')

plt.show()
