def normal_equation(data):
        
        X, Y = data.to_matrix(degree=1)
        
        X_transpose = Matrix(X).transpose().data
        X_transpose_X = Matrix(X_transpose).multiply(Matrix(X)).data
        X_transpose_Y = Matrix(X_transpose).multiply(Matrix([Y])).data
        
        theta = Matrix(X_transpose_X).gaussian_elimination([x[0] for x in X_transpose_Y])
        
        return theta[1], theta[0]
