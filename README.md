# Numpy

The core concept of numpy arrays

<p align="center">
 <img src="https://img.shields.io/static/v1?label=language&message=python&color=green"/>
<img src="https://img.shields.io/static/v1?label=liked-most&message=jupyter-notebook&color=brightgreen"/>
<img src="https://img.shields.io/static/v1?label=liked-most&message=numpy&color=blueviolet"/>
</p>

<p align="center">
  <img src="https://github.com/CrispenGari/Numpy/blob/main/download.png"/>
</p>

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

```python
import numpy as np

```

## Creating arrays

```python
# Createing a 1d array

a = np.array([2,3,4,5,6])

print(a)

# Creating a 2d array

b = np.array([[1,2,3,4], [5,6,7,8]])
print(b)

```

## Getting dimensions of arrays

```python
# Getting the dimension of the array
print(a.ndim)
print(b.ndim)
```

## Getting the shape of the array

```python
# Getting the shape of numpy arrays
print(a.shape)
print(b.shape) # returns (2 rows, 4 colms)
```

## Getting the data type of the array

```python
# Getting the datatype of the array in numpy
print(a.dtype) # int32
```

### Specifying the data type of the array during initialization

```python
# Specifing the datatype in a numpy array

c = np.array([2, 3,4,5,6,8,9], dtype='int16')

# getting the datatype of numpy array c

print(c.dtype) # int16
```

## Getting the size of the array

```python
# Getting the size of the array
# int32 type
print(a.itemsize) # 4
# the int16 type
print(c.itemsize) # 2
```

## Getting the total number of elements in an array

```python
#  Getting the total number of elements in the array
print(a.size) # 4
print(b.size) # 8
```

## Getting the total size of an array

```python
## Getting the total size of an array

## ------------ METHOD 1
print(a.itemsize * a.size)
##------------- METHOD 2
print(a.nbytes)
```

## Array Indexing

### Getting the a specific element in a a given position in a N-d array

```python
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

```python
# GETTING A SPECIFIC COLUMN (last column)
print(npArray[:, -1])
print(npArray[:, 5])

```

### Getting a specific row in an n-d array

```python
# GETTING A SPECIFIC ROW
print(npArray[1, :])
# OR
print(npArray[-1: :][0])
```

### Numpy array indexing using

`array[index, startindex, endindex, step]`

```python
### Numpy array indexing using `array[index, startindex, endindex, step]`

# Let's say we want to get all the numbers that ends with 5

print(npArray[0, 0: -1: 2])
```

### Changing the item value in an array

`array[index, index] = value`

```python
## Changing the item value in an array `array[index, index] = value`

# Let's say we ant to update 10 in the npArray to be 100

npArray[1, -2] = 100
print(npArray[1,-2]) # 100

# Let's say now we want to change all the numbers that ends with 5 to have a value of 99
npArray[0, 0: -1:2] = 99

npArray
```

### 3 Dimensional Array

```python

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

```python
### np.zeros(shape, type)
zeros = np.zeros([2, 4, 6])
zeros
```

### np.ones(shape, type)

```python
### np.ones(shape, type)
ones = np.ones((2, 3,5), dtype=np.uint32)
ones
```

### np.full(shape, number, dtype)

```python
### any number 255
whiteImage = np.full((4,4), 255, dtype='uint32')
whiteImage
```

### np.full_like(shape_of_the_existing_array, number)

```python
### full_like
# Let's say we want to generate the array of 100s that has the same shape with whiteImage array

array100s = np.full_like(whiteImage, 9)
array100s

```

## Generating array from random numbers in numpy

### np.random.rand(shape)

```python
randomFloat = np.random.rand(2,3,6)
randomFloat

```

### np.random.randint(start, end_exclusive, size)

```python

# Generating an array of integers
randomInt = np.random.randint(-7, 15, size=(7, 9,7))
randomInt

```

### np.identity(number, dtype)

Generates the identity matrix

```python
identity = np.identity(4, dtype='int8')
identity
```

### Generating array of integers using `np.arange(n).reshape(shape)`

> Note that the product of the shape (a, b) must give us n which means `n = a * b` otherwise an error will occur the following is an example of how we can generate array of integers using the two methods `arange` and `reshape`

```python
## Generating element's using the np.arange(15).reshape(3, 5) methods
arr = np.arange(15).reshape(3,5)
arr
```

## Repeating an array

### Repeating an array vertically

```python
arr1 = np.array([[2,4,6]])
arr2 = np.repeat(arr1, 3, axis=0) #  repeats vertically
arr2
```

### Repeating an array horizontally

```python
arr3 = np.repeat(arr1, 3, axis=1) # Repeats horizontally
arr3
```

### Universal Functions

- arange(n)

```python
# the arange(n) function that generates 1-d array of integers from 0 to n-1
a = np.arange(10)
a # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

```

> There are a lot of functions that are used in numpy arrays some of them are

- all
  - Test whether all array elements along a given axis evaluate to True.
  ```python
  print(np.all(np.array([True, False, True, True, False]))) # False
  print(np.all(np.array([True, True]))) # True
  print(np.all(np.array([1,2, 3,-10]))) # True
  ```
- any
  - Test whether any array element along a given axis evaluates to True.
  ```python
  print(np.any(np.array([True, False, True, True, False]))) # True
  print(np.any(np.array([not True, not True]))) # False
  print(np.any(np.array([1,2, 3,-10]))) # True
  ```
- apply_along_axis
  - Read more [here](https://numpy.org/doc/1.14/reference/generated/numpy.apply_along_axis.html#numpy.apply_along_axis)
- argmax

  - Returns the indices of the maximum values along an axis.

  ```python
  print(np.argmax(np.array([1,10,-19,9, 0,56,108,-76]))) # 6
  ```

- argmin

  - Returns the indices of the minimum values along an axis.

  ```python
  print(np.argmax(np.array([1,10,-19,9, 0,56,108,-76]))) # 7
  ```

- argsort
  - Returns the indices that would sort an array.
  ```python
  print(np.argmax(np.array([1,10,-19,9, 0,56,108,-76]))) # 7
  ```
- average
  - Compute the weighted average along the specified axis.
  ```python
  print(np.average(np.array([1,10,-19,9, 0,56,108,-76]))) # 11.125
  ```
- bincount
  - Count number of occurrences of each value in array of non-negative ints.

```python
# bincount
print(np.bincount(np.array([1,1,2,6,7,0]))) # [1 2 1 0 0 0 1 1]
```

- ceil
  - Return the ceiling of the input, element-wise from floats values

```python
# ceil function

print(np.ceil(np.array([1.2, 2.8, 6.9, -6,10,9]))) # [ 2.  3.  7. -6. 10.  9.]
```

- clip
  - Clip (limit) the values in an array.

```python
# clip function
print(np.clip(np.arange(10), 2, 8)) # [2 2 2 3 4 5 6 7 8 8]
```

- conj
  - Return the complex conjugate, element-wise.
    [Example](https://numpy.org/doc/1.14/reference/generated/numpy.conj.html#numpy.conj)
- corrcoef
  - Return Pearson product-moment correlation coefficients.
    [Example](https://numpy.org/doc/1.14/reference/generated/numpy.corrcoef.html#numpy.corrcoef)
- cov
  - Estimate a covariance matrix, given data and weights.
    [Example](https://numpy.org/doc/1.14/reference/generated/numpy.cov.html#numpy.cov)
- cross
  - Return the cross product of two (arrays of) vectors.
    [Example](https://numpy.org/doc/1.14/reference/generated/numpy.cross.html#numpy.cross)
- cumprod
  - Return the cumulative product of elements along a given axis.

```python
print(np.cumprod(np.array([2,3,5]))) # [ 2  6 30]
```

- cumsum
  - Return the cumulative sum of elements along a given axis.

```python
print(np.cumsum(np.array([2,3,5]))) # [ 2  5 10]
```

- diff

  - Calculate the n-th discrete difference along the given axis.
    [Example](https://numpy.org/doc/1.14/reference/generated/numpy.diff.html#numpy.diff)

- dot
  - Dot product of two arrays. Specifically
    [Example](https://numpy.org/doc/1.14/reference/generated/numpy.dot.html#numpy.dot)
- floor

```python
# floor function
print(np.floor(np.array([1.2, 2.8, 6.9, -6,10,9]))) # [ 1.  2.  6. -6. 10.  9.]
```

- inner

  - Inner product of two arrays.
    [Example](https://numpy.org/doc/1.14/reference/generated/numpy.inner.html#numpy.inner)

- lexsort
  - Perform an indirect sort using a sequence of keys.
    [Example](https://numpy.org/doc/1.14/reference/generated/numpy.lexsort.html#numpy.lexsort)
- max
  - Return the largest item in an iterable or the largest of two or more arguments.

```python
# max function
print(np.max(np.random.randint(1, 100, 5))) # a random numbaer
```

- maximum
  - Element-wise maximum of array elements.
  ```python
  print (np.maximum([2, 3, 4], [1, 5, 2])) # [2 5 4]
  ```
- mean
  - Compute the arithmetic mean along the specified axis.

```python
# mean
print(np.mean(np.ones((5,)))) # 1.0
```

- median
  - Compute the median along the specified axis.

```python
print(np.median(np.array([2, 5, -6, 8, 9]))) # 5
```

- min
  - Return the smallest item in an iterable or the largest of two or more arguments.

```python
# max function
print(np.min(np.random.randint(1, 100, 5))) # a random number
```

- ## minimum

```python
print (np.minimum([2, 3, 4], [1, 5, 2])) # [1 3 2]
```

- nonzero
  - Return the indices of the elements that are non-zero.

```python
# nonzero function
print(np.nonzero(np.round(np.array(np.random.rand(10))))) # (array([2, 3, 4, 6, 7, 8], dtype=int64),)
```

- outer
  - Compute the outer product of two vectors.
    [Example](https://numpy.org/doc/1.14/reference/generated/numpy.outer.html#numpy.outer)
- prod
  - Return the product of array elements over a given axis.

```python
# prod function
print(np.prod(np.array([2,2,2,3]))) # 24
```

- re
  - This module provides regular expression matching operations similar to those found in Perl.
    [Example](https://docs.python.org/dev/library/re.html#module-re)
- round
  - Return number rounded to ndigits precision after the decimal point. If ndigits is omitted or is None, it returns the nearest integer to its input.

```python
np.round(np.array(np.random.rand(10))) # a random array
```

- sort
  - Return a sorted copy of an array.

```python
np.sort(np.array([1, 2,10, 0,7,-6])) # array([-6,  0,  1,  2,  7, 10])
```

- std
  - Compute the standard deviation along the specified axis.

```python
np.std(np.array([5, 9, 9, 0])) # 3.6996621467371855
```

- sum
  - Sum of array elements over a given axis

```python
np.sum(np.ones(5)) # 5
```

- trace
  - Return the sum along diagonals of the array.
  ```python
  A = np.array([
   [2, 3, 4.],
   [4, 5, 9]
  ])
  np.trace(A) # 7
  ```
- transpose
  - Permute the dimensions of an array.
  ```python
  A = np.array([
   [2, 3, 4.],
   [4, 5, 9]])
   print(np.transpose(A))
  ```
- var
  - Compute the variance along the specified axis.

```python
# var function the varience of the array
np.var(np.array([5, 9, 9, 0])) # 13.6875
```

- vdot
  - Return the dot product of two vectors.
    [Example](https://numpy.org/doc/1.14/reference/generated/numpy.vdot.html#numpy.vdot)
- vectorize
  - Generalized function class.
    [Example](https://numpy.org/doc/1.14/reference/generated/numpy.vectorize.html#numpy.vectorize)
- where
  Return elements, either from x or y, depending on condition.
  [Example](https://numpy.org/doc/1.14/reference/generated/numpy.where.html#numpy.where)

## Copying numpy arrays

### Direct copying

> This method just assigns the first array to the second array. So when the value of the first second array changes then the value of the first array also changes.

```python

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

```python

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

```python

a = np.array([2,3,4,5,9,90])
b = np.array([1,2,3,4,5,6])

```

### Trigonometry

```python

### Trigonometry

print(np.sin(a))
print(np.cos(a))
print(np.tan(a))

# There are more

```

### Maths on two arrays

```python

### Maths on two arrays

print(a * b)
print(a ** b)
print(a / b)
print(a - b)
print(a + b)

# There are more

```

### Math on one array

```python

### Math on one array

print(a \* 3)
print(a \*\* 3)
print(a / 3)
print(a - 3)
print(a + 3)

# There are more

```

## Linear Algebra `dot products`

> In Linear Algebra, we are more focusing on multiplying arrays. So the rule is simple, THE NUMBER OF ONE COLUMNS OF MATRIX a SHOULD BE EQUAL TO THE NUMBER OF ROWS OF MATRIX B.

```python

a = np.ones([2, 2], dtype='int32')
b = np.full([2,2], 3, dtype= 'int32')

print(a \* b)

# The correct way

print(np.matmul(a, b))

## Getting the deteminant of a

print(np.linalg.det(b))

```

> For more of these on linear algebra go to the [Docs](https://docs.scipy.org/doc/)

## Statistics

Suppose we have the following array.

```python

arr = np.array([[2, 3, 4, 5,-1,9],[100, 19, 100, 78,10,-90]])

```

### Finding the maximum element of the array

```python

print(np.max(arr))

```

### Finding the minimum element of the array

```python

print(np.min(arr))

```

### Summing the elements of the array

```python

print(np.sum(arr))

```

## Reshaping arrays

`arr.reshape(dimension)`
Suppose we have an array of numbers. And we want to create another array from this array with different dimension. We can use the reshape method to reshape the array. Example:

```python

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

```python

arr1 = np.array([2,3,4,5,8])
arr2 = np.array([1,2,3,4,5])
print(np.hstack([arr1, arr2]))

```

### vertical stack of arrays

```python

arr1 = np.array([2,3,4,5,8])
arr2 = np.array([1,2,3,4,5])
print(np.vstack([arr1, arr2]))

```

## Generating arrays from a file

> Suppose we have a file that has list of numbers seperated by a comma and we want to read this file and into a numpy array. The file name is 'data.txt'. This can be done as follows

```python

data = np.genfromtxt('data.txt', delimiter=',')
data # the data in the file as float type

data = np.genfromtxt('data.txt', delimiter=',', dtype='int32')
data # the data in the file as int32

# As integer datatype

data = np.genfromtxt('data.txt', delimiter=',', dtype='int32')
data

# Or

data.astype('int8')

```

## Boolean Masking and Advanced indexing.

> Gives us the ability to mask the elements to boolean based on certain conditions

> Example: Checking if the element at that index is even or not.

```python

data = np.genfromtxt('data.txt', delimiter=',', dtype='int32')
data % 2 == 0

```

> Example: returning odd elements from the array

```python

data = np.genfromtxt('data.txt', delimiter=',', dtype='int32')
data[data%2==1]

```

> Example: Indexing multiple elements
> Let's say we want to index and return elements that are at index, `1, 2, 6, 9` and the last element in the array `data` where elements are `ODD` we can do it as follows:

```python
data = np.genfromtxt('data.txt', delimiter=',', dtype='int32')
data[data%2==1][1, 2, 6, 9, -1]] # array([ 3, 5, 13, 19, 79])

```

## Where to find the documentation?

- [Documentation](https://numpy.org/doc/)

- [Documentation](https://numpy.org/doc/1.14/user/quickstart.html)
