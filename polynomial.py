def polynomial(self, degree=2):
  
        def generate_polynomial_features(x):
                return [x**i for i in range(degree + 1)]
        
        X = np.array([generate_polynomial_features(point[0]) for point in self.data])
        
        Y = np.array([point[1] for point in self.data])
        
        XT_X = X.T.dot(X)
        XT_Y = X.T.dot(Y)
        
        XT_X = XT_X.tolist()
        XT_Y = XT_Y.tolist()
        
        theta = self.gaussian_elimination(XT_X, XT_Y)
        
        return theta
