# Numpy

The core concept of numpy arrays

## Why numpy over python list

- numpy are faster because the used fixed type
- List requires more memory than numpy arrays

## Applications

- Mathematics (MATLAB)
- Backend
- Plotting
- Machine Learning

## Getting started

Install numpy using pip by running the following command

```shell
$ pip install numpy
OR
$ python -m pip install numpy

```

## Importing numpy to our project

```
import numpy as np

```

## Creating arrays

```
# Createing a 1d array

a = np.array([2,3,4,5,6])

print(a)

# Creating a 2d array

b = np.array([[1,2,3,4], [5,6,7,8]])
print(b)

```

## Getting dimensions of arrays

```
# Getting the dimension of the array
print(a.ndim)
print(b.ndim)
```

## Getting the shape of the array

```
# Getting the shape of numpy arrays
print(a.shape)
print(b.shape) # returns (2 rows, 4 colms)
```

## Getting the data type of the array

```
# Getting the datatype of the array in numpy
print(a.dtype) # int32
```

### Specifying the data type of the array during initialization

```
# Specifing the datatype in a numpy array

c = np.array([2, 3,4,5,6,8,9], dtype='int16')

# getting the datatype of numpy array c

print(c.dtype) # int16
```

## Getting the size of the array

```
# Getting the size of the array
# int32 type
print(a.itemsize) # 4
# the int16 type
print(c.itemsize) # 2
```

## Getting the total number of elements in an array

```
#  Getting the total number of elements in the array
print(a.size) # 4
print(b.size) # 8
```

## Getting the total size of an array

```
## Getting the total size of an array

## ------------ METHOD 1
print(a.itemsize * a.size)
##------------- METHOD 2
print(a.nbytes)
```

## Array Indexing

### Getting the a specific element in a a given position in a N-d array

```
npArray = np.array([[5, 10, 15, 20, 25, 30], [2, 4,6, 8, 10, 12]])
# The shape
print(npArray.shape)

# The number of elements
print(npArray.size)

# The total memory in an array
print(npArray.size * npArray.itemsize)
# OR
print(npArray.nbytes)


# GETTING THE ELEMENT 10 IN OUR ARRAY

print(npArray[1, -2])
# OR
print(npArray[1, 4])

```

### Getting a specific column in an n-d array

```
# GETTING A SPECIFIC COLUMN (last column)
print(npArray[:, -1])
print(npArray[:, 5])

```

### Getting a specific row in an n-d array

```
# GETTING A SPECIFIC ROW
print(npArray[1, :])
# OR
print(npArray[-1: :][0])
```

### Numpy array indexing using

`array[index, startindex, endindex, step]`

```
### Numpy array indexing using `array[index, startindex, endindex, step]`

# Let's say we want to get all the numbers that ends with 5

print(npArray[0, 0: -1: 2])
```

### Changing the item value in an array

`array[index, index] = value`

```
## Changing the item value in an array `array[index, index] = value`

# Let's say we ant to update 10 in the npArray to be 100

npArray[1, -2] = 100
print(npArray[1,-2]) # 100

# Let's say now we want to change all the numbers that ends with 5 to have a value of 99
npArray[0, 0: -1:2] = 99

npArray
```

### 3 Dimensional Array

```

# 3 Dimensional Array

np3D = np.array([
    [
        [2,4,6,8,10],
        [4,8,12,16,20]
    ],
    [
        [3,6,9, 12, 15],
        [6,12,18, 24, 30]
    ]
])

#  Getting the dimension

print(np3D.ndim) # 3

# Getting the total elements
print(np3D.size) # 20

# Getting the total memory
print(np3D.nbytes) # 80

# Getting the element 18 from the array
print(np3D[-1,-1, -3]) # 18

```

## Initializing Different types of arrays Matrix

### np.zeros(shape, type)

```
### np.zeros(shape, type)
zeros = np.zeros([2, 4, 6])
zeros
```

### np.ones(shape, type)

```
### np.ones(shape, type)
ones = np.ones((2, 3,5), dtype=np.uint32)
ones
```

### np.full(shape, number, dtype)

```
### any number 255
whiteImage = np.full((4,4), 255, dtype='uint32')
whiteImage
```

### np.full_like(shape_of_the_existing_array, number)

```
### full_like
# Let's say we want to generate the array of 100s that has the same shape with whiteImage array

array100s = np.full_like(whiteImage, 9)
array100s

```

## Generating array from random numbers in numpy

### np.random.rand(shape)

```
randomFloat = np.random.rand(2,3,6)
randomFloat

```

### np.random.randint(start, end_exclusive, size)

```

# Generating an array of integers
randomInt = np.random.randint(-7, 15, size=(7, 9,7))
randomInt

```

### np.identity(number, dtype)

Generates the identity matrix

```
identity = np.identity(4, dtype='int8')
identity
```

## Repeating an array

### Repeating an array vertically

```
arr1 = np.array([[2,4,6]])
arr2 = np.repeat(arr1, 3, axis=0) #  repeats vertically
arr2
```

### Repeating an array horizontally

```
arr3 = np.repeat(arr1, 3, axis=1) # Repeats horizontally
arr3
```

## Copying numpy arrays

### Direct copying

> This method just assigns the first array to the second array. So when the value of the first second array changes then the value of the first array also changes.

```
## Copying arrays
array1 = np.array([1,2,3,4,5])
array2 = array1
print(array2) # [1 2 3 4 5]
array2[-1]= 500
print(array2) # [1 2 3 4 500]
# Array1 also changes
print(array1) # [1 2 3 4 500]
```

### using the copy() method

> This method copies the array content not the original array. So the content of the copied array when changed doesn't affects the content of the original array.

```
array1 = np.array([1,2,3,4,5])
array2 = array1.copy()
print(array2) # [1 2 3 4 5]
array2[-1]= 500
print(array2) # [1 2 3 4 500]
# Array1 also changes
print(array1) # [1 2 3 4 5]
```

### Mathematics on numpy arrays

Suppose we have the following arrays

```
a = np.array([2,3,4,5,9,90])
b = np.array([1,2,3,4,5,6])
```

### Trigonometry

```
### Trigonometry
print(np.sin(a))
print(np.cos(a))
print(np.tan(a))
#  There are more

```

### Maths on two arrays

```
### Maths on two arrays

print(a * b)
print(a ** b)
print(a / b)
print(a - b)
print(a + b)
# There are more
```

### Math on one array

```
### Math on one array
print(a * 3)
print(a ** 3)
print(a / 3)
print(a - 3)
print(a + 3)
# There are more
```

## Linear Algebra

> In Linear Algebra, we are more focusing on multiplying arrays. So the rule is simple, THE NUMBER OF ONE COLUMNS OF MATRIX a SHOULD BE EQUAL TO THE NUMBER OF ROWS OF MATRIX B.

```
a = np.ones([2, 2], dtype='int32')
b = np.full([2,2], 3, dtype= 'int32')

print(a * b)
# The correct way
print(np.matmul(a, b))
## Getting the deteminant of a
print(np.linalg.det(b))

```

> For more of these on linear algebra go to the [Docs](https://docs.scipy.org/doc/)

## Statistics

Suppose we have the following array.

```
arr = np.array([[2, 3, 4, 5,-1,9],[100, 19, 100, 78,10,-90]])
```

### Finding the maximum element of the array

```
print(np.max(arr))
```

### Finding the minimum element of the array

```
print(np.min(arr))

```

### Summing the elements of the array

```

print(np.sum(arr))
```

## Reshaping arrays

`arr.reshape(dimension)`
Suppose we have an array of numbers. And we want to create another array from this array with different dimension. We can use the reshape method to reshape the array. Example:

```
arr = np.array([[2, 3, 4,5],[100, 19, 100, 7]])
print(arr.shape) # (2,4)

arr2 = arr.reshape([1, 8])
print(arr2.shape) # (1, 8)

```

> When reshaping array all you need to take care of is the elements that are in the array meaning the number of elements in an array must be equal to the product of the dimension of the array. For example `1 * 8 = 8` elements which are the elements that are in the first array

## Stacking arrays

Arrays can be stacked together to produce 1 array using two numpy methods

- `np.vstack([arr1, arr2])` Stacks arrays vertically
- `np.hstack([arr1, arr2])` Stacks array horizontally

### horizontal Stack of arrays

```
arr1 = np.array([2,3,4,5,8])
arr2 = np.array([1,2,3,4,5])
print(np.hstack([arr1, arr2]))
```

### vertical stack of arrays

```
arr1 = np.array([2,3,4,5,8])
arr2 = np.array([1,2,3,4,5])
print(np.vstack([arr1, arr2]))
```

## Generating arrays from a file

> Suppose we have a file that has list of numbers seperated by a comma and we want to read this file and into a numpy array. The file name is 'data.txt'. This can be done as follows

```
data = np.genfromtxt('data.txt', delimiter=',')
data # the data in the file as float type

data = np.genfromtxt('data.txt', delimiter=',', dtype='int32')
data # the data in the file as int32

# As integer datatype

data = np.genfromtxt('data.txt', delimiter=',', dtype='int32')
data
#  Or
data.astype('int8')

```

## Boolean Masking and Advanced indexing.

> Gives us the ability to mask the elements to boolean based on certain conditions

> Example: Checking if the element at that index is even or not.

```
data = np.genfromtxt('data.txt', delimiter=',', dtype='int32')
data % 2 == 0
```

> Example: returning odd elements from the array

```
data = np.genfromtxt('data.txt', delimiter=',', dtype='int32')
data[data%2==1]
```

> Example: Indexing multiple elements
> Let's say we want to index and return elements that are at index, `1, 2, 6, 9` and the last element in the array `data` where elements are `ODD` we can do it as follows:

```
data = np.genfromtxt('data.txt', delimiter=',', dtype='int32')
data[data%2==1][[1, 2, 6, 9, -1]] # array([ 3,  5, 13, 19, 79])

```

## Where to find the documentation?

- [Documentation](https://numpy.org/doc/)
