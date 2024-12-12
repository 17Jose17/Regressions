import numpy as np
import math

class Data:
        def __init__(self, data):
                self.raw_data = data
                self.data = np.array(data)
                self.features = self.data[:, :-1] if len(data[0]) > 1 else None
                self.targets = self.data[:, -1] if len(data[0]) > 1 else None
        
        def normalize(self):
                if self.features is not None:
                        min_vals = self.features.min(axis=0)
                        max_vals = self.features.max(axis=0)
                        self.features = (self.features - min_vals) / (max_vals - min_vals)
                        self.data[:, :-1] = self.features
        
        def split(self, test_size=0.2, shuffle=True):
                if shuffle:
                        np.random.shuffle(self.data)
            
                split_index = int(len(self.data) * (1 - test_size))
                train_data = self.data[:split_index]
                test_data = self.data[split_index:]
                return train_data, test_data
        
        def to_matrix(self, degree=1):
                if self.features is not None:
                        X = np.array([[x ** i for i in range(degree + 1)] for x in self.features[:, 0]])
                        Y = self.targets
                        return X, Y
                return None, None

class Matrix:
        def __init__(self, data):
                self.data = data
                self.filas = len(self.data)
                self.columnas = len(self.data[0])
        
        def transpose(self):
                return Matrix([[self.data[j][i] for j in range(self.filas)] for i in range(self.columnas)])
    
        def multiply(self, otra_matriz):
                if self.columnas != otra_matriz.filas:
                        raise ValueError("Las matrices no son multiplicables: el número de columnas de la primera debe ser igual al número de filas de la segunda.")
            
            resultado = [[0] * otra_matriz.columnas for _ in range(self.filas)]
            
            for i in range(self.filas):
                    for j in range(otra_matriz.columnas):
                            for k in range(self.columnas):
                                    resultado[i][j] += self.data[i][k] * otra_matriz.data[k][j]
            
            return Matrix(resultado)

        def determinant(self):
                if self.filas != self.columnas:
                        raise ValueError("El determinante solo está definido para matrices cuadradas.")
            
            A = [row[:] for row in self.data]
            n = self.filas
            det = 1
            for i in range(n):
                    max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
                    if i != max_row:
                            A[i], A[max_row] = A[max_row], A[i]
                            det *= -1
                
                if A[i][i] == 0:
                        return 0
                
                det *= A[i][i]
                for j in range(i + 1, n):
                        factor = A[j][i] / A[i][i]
                        for k in range(i, n):
                                A[j][k] -= factor * A[i][k]
            
            return det

        def gaussian_elimination(self, b):
            n = len(b)
            A = [row[:] for row in self.data]
            b = b[:]
            
            for i in range(n):
                    if A[i][i] == 0:
                            raise ValueError("El pivote es cero, no se puede resolver con eliminación gaussiana.")
                    for j in range(i + 1, n):
                            factor = A[j][i] / A[i][i]
                            for k in range(i, n):
                                    A[j][k] -= factor * A[i][k]
                            b[j] -= factor * b[i]
            
            x = [0] * n
            for i in range(n - 1, -1, -1):
                    suma = sum(A[i][j] * x[j] for j in range(i + 1, n))
                    x[i] = (b[i] - suma) / A[i][i]
            
            return x
    
        def show(self):
                for fila in self.data:
                        print(fila)
