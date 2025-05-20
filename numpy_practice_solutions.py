# NumPy Practice Assignment Solutions
import numpy as np

# 1. Create a 1D NumPy array with values from 10 to 50
arr1 = np.arange(10, 51)
print("1. 1D Array from 10 to 50:\n", arr1)

# 2. Create a 2D array with 3 rows and 4 columns filled with ones
arr2 = np.ones((3, 4))
print("\n2. 2D Array (3x4) with ones:\n", arr2)

# 3. Generate 5 random float numbers between 0 and 1 using np.random.rand()
rand_floats = np.random.rand(5)
print("\n3. Random floats:\n", rand_floats)

# 4. Generate a 3x3 matrix of random integers between 1 and 20
rand_ints = np.random.randint(1, 21, size=(3, 3))
print("\n4. 3x3 Random Integer Matrix (1-20):\n", rand_ints)

# 5. Create an array of 6 zeros and reshape it into 2 rows and 3 columns
zeros_reshaped = np.zeros((2, 3))
print("\n5. 2x3 Array of Zeros:\n", zeros_reshaped)

# 6. Create an identity matrix of size 4x4
identity_matrix = np.eye(4)
print("\n6. Identity Matrix (4x4):\n", identity_matrix)

# 7. Create an array of numbers from 0 to 20 with a step of 5
step_array = np.arange(0, 21, 5)
print("\n7. Array with step of 5 (0 to 20):\n", step_array)

# 8. Use slicing to extract elements 2 to 4 from this array: np.array([5, 10, 15, 20, 25, 30])
slice_array = np.array([5, 10, 15, 20, 25, 30])
sliced = slice_array[1:4]
print("\n8. Sliced Elements (index 1 to 3):\n", sliced)

# 9. Access the value 8 from this 2D array: np.array([[4, 8], [16, 32]])
arr_2d = np.array([[4, 8], [16, 32]])
value = arr_2d[0, 1]
print("\n9. Value accessed (8):\n", value)

# 10. Generate 10 evenly spaced values between 0 and 100 using np.linspace()
linspace_vals = np.linspace(0, 100, 10)
print("\n10. 10 Evenly Spaced Values (0 to 100):\n", linspace_vals)
