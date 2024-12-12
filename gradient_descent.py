def gradient_descent(data, lr=0.0001, epochs=100000, tolerance=1e-6):
        
        X, Y = data.to_matrix()
      
        m, b = 0, 0
        n = len(X)
    
        for _ in range(epochs):
                Y_pred = [m * x + b for x in X[:, 0]]
        
                X_matrix = Matrix(X)
                Y_matrix = Matrix([Y])
                Y_pred_matrix = Matrix([Y_pred])
        
                dm = 2 / n * X_matrix.multiply(Y_pred_matrix.resta(Y_matrix)).data
                db = 2 / n * np.sum(Y_pred_matrix - Y_matrix)
                
                m_prev, b_prev = m, b
                
                m -= lr * dm
                b -= lr * db
                
                if abs(m - m_prev) < tolerance and abs(b - b_prev) < tolerance:
                        break
        
        return m, b
