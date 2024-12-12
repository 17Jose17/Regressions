def multivariate(self):

        X = np.array([[1, point[0], point[1]] for point in self.data])
        
        Y = np.array([point[2] for point in self.data])
        
        XT_X = X.T.dot(X)
        XT_Y = X.T.dot(Y)
        
        XT_X = XT_X.tolist()
        XT_Y = XT_Y.tolist()
        
        theta = self.gaussian_elimination(XT_X, XT_Y)
        
        return theta
