import numpy as np

# Pauli Y
Y = np.array([[0, -np.complex(0,1)],
             [np.complex(0,1), 0]])

# Pauli Z 
Z = np.array([[1,0], [0,-1]])

# Computing thhe Kronecerkker Product

kron_YZ = np.kron(Y,Z)

print(kron_YZ)    