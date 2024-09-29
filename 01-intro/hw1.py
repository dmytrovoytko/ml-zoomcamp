import pandas as pd
print(pd.__version__)

file_name = 'laptops.csv'
df = pd.read_csv(file_name, index_col=False)

### Q2. Records count
print()
print('rows:', df.shape[0])

### Q3. Laptop brands
print()
print('Brand.nunique =', df.Brand.nunique())

### Q4. Missing values
print()
print('info')
print(df.info())
print()
isnull = df.isnull().sum()
print(isnull)
null_cols = df.columns[df.isnull().any()].tolist()
print('Columns with missing values:', null_cols, len(null_cols))

### Q5. Maximum final price
print()
print('Dell Final Price')
df_dell = df[df.Brand=='Dell']
print(df_dell['Final Price'].describe())
print('Max', df_dell['Final Price'].max())

### Q6. Median value of Screen
print()
screen_median = df['Screen'].median()
screen_mode = df['Screen'].mode()[0]
print('Screen Median =', screen_median, 'Mode =', screen_mode)
df['Screen'] = df['Screen'].fillna(screen_mode)
print('Check null columns?', df.columns[df.isnull().any()].tolist())
screen_median_new = df['Screen'].median()
print('Median changed:', screen_median_new != screen_median)

### Q7. Sum of weights
print()
# 1. Select all the "Innjoo" laptops from the dataset.
# 2. Select only columns `RAM`, `Storage`, `Screen`.
df_Innjoo = df[df.Brand=='Innjoo'][['RAM', 'Storage', 'Screen']]

import numpy as np

# 3. Get the underlying NumPy array. Let's call it `X`.

X = np.array(df_Innjoo)
print(X)


# 4. Compute matrix-matrix multiplication between the transpose of `X` and `X`. To get the transpose, use `X.T`. Let's call the result `XTX`.

def vector_vector_multiplication(u, v):
    assert u.shape[0] == v.shape[0]
    
    n = u.shape[0]
    
    result = 0.0

    for i in range(n):
        result = result + u[i] * v[i]
    
    return result

def matrix_vector_multiplication(U, v):
    assert U.shape[1] == v.shape[0]
    
    num_rows = U.shape[0]
    
    result = np.zeros(num_rows)
    
    for i in range(num_rows):
        result[i] = vector_vector_multiplication(U[i], v)
    
    return result

def matrix_matrix_multiplication(U, V):
    assert U.shape[1] == V.shape[0]
    
    num_rows = U.shape[0]
    num_cols = V.shape[1]
    
    result = np.zeros((num_rows, num_cols))
    
    for i in range(num_cols):
        vi = V[:, i]
        Uvi = matrix_vector_multiplication(U, vi)
        result[:, i] = Uvi
    
    return result        

XTX = matrix_matrix_multiplication(X.T, X)
print(XTX)

# 5. Compute the inverse of `XTX`.

inv = np.linalg.inv(XTX)
print(inv)

# 6. Create an array `y` with values `[1100, 1300, 800, 900, 1000, 1100]`.
y = np.array([1100, 1300, 800, 900, 1000, 1100])

# 7. Multiply the inverse of `XTX` with the transpose of `X`, and then multiply the result by `y`. Call the result `w`.

w_ = matrix_matrix_multiplication(inv, X.T)
print(w_)
w = matrix_vector_multiplication(w_, y)
print('w:', w)

# 8. What's the sum of all the elements of the result?
print('sum = ', np.sum(w))



