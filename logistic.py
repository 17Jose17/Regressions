def logistic(data, lr=0.01, epochs=10000, tolerance=1e-6):
  
        def sigmoid(z):
                return 1 / (1 + np.exp(-z))
          
        def calculate_norm(vector):
                return np.sqrt(np.sum(vector ** 2))
    
        X, Y = data.to_matrix()
    
        m, n = X.shape
        theta = np.zeros(n)
    
        for _ in range(epochs):
                Y_pred = sigmoid(np.dot(X, theta))
        
                gradient = (1 / m) * np.dot(X.T, (Y_pred - Y))
        
                theta -= lr * gradient
        
                if calculate_norm(gradient) < tolerance:
                        break
    
        return theta
