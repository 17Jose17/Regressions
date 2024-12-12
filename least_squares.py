def least_squares(data):
  
        X, Y = data.to_matrix()
        
        X_transpose = Matrix(X).transpose().data
        
        X_transpose_X = Matrix(X_transpose).multiply(Matrix(X)).data
        
        X_transpose_Y = Matrix(X_transpose).multiply(Matrix([Y])).data
        
        theta = Matrix(X_transpose_X).gaussian_elimination([x[0] for x in X_transpose_Y])
        
        return theta[0], theta[1]
