#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Create by python sequence : np.array(python_seq)
import numpy as np
list1 = range(1,5)
nparray = np.array(list1)
print(nparray)
tuple = (2,1,5,'s',6)
nparray2 = np.array(tuple)
print(nparray2)


# In[2]:


# Creating by Numpy method np.arange()
A = np.arange(1,6)
print(A)

B = np.arange(1,6,2)
print(B)

C = np.arange(6,1,-1)
print(C)


# In[3]:


# Create by Pattern : np.zeros(n) [np.array(python_seq*n)]
import numpy as np
x = (1,2)
numpy_array = np.array(x*3)
print(numpy_array)
numpy_arrays = np.zeros(3)
print(numpy_arrays)
y = np.array(np.array(x)*3)
print(y)


# In[4]:


# Create by converting from other objects [np.asarray(object)]
z = range(1,5) # now z is a python list
np.asarray(z) # convert z into a np array
print(z)


# In[5]:


# Create Numpy Arrays using 'dtype='
A = np.array([1,2,3,5,7], dtype = np.int32)
B = np.array([1,2,3,5,7], dtype = np.float)
C = np.array([1,2,3,5,7], dtype = np.str)
D = np.array([1,2,3,5,7], dtype = np.bool)
print(A,B,C,D)
print(A.dtype,B.dtype,C.dtype,D.dtype)


# In[6]:


# Multidemensional Numpy Arrays
A = np.array([[2,0],[1,0],[0,0]])
print(A)


# In[7]:


S = np.array([np.arange(2),np.arange(2),['a','b']])
print(S)
S.shape


# # Operations on Numpy Array

# ## Mathematicl Operations
# ### Different b/w Python list and numpy array 

# In[8]:


np.arange(3)+np.arange(3)


# In[9]:


np.arange(3)+2


# In[10]:


np.arange(3)-np.arange(3)


# In[11]:


np.arange(1,5)/(np.zeros(4)+2)


# In[12]:


np.array([1,2,4])*np.array([1,1,2])


# # Properties of Numpy Array

# In[13]:


# A.ndim--retuen the number of NP array dimensions
A = np.array([range(1,5),range(2,6)])
A.ndim


# In[14]:


# A.shape--return the number of elements in each dimension
A = np.array([range(1,5),range(2,6)])
A.shape


# In[15]:


# A.T--return a transposed Numpy Array if the # of array dimensions more than one otherwise return A self
A = np.array([range(1,5),range(2,6)])
B = A.T
print(B)


# In[16]:


# A.dtype--retuen the data type of NP array's elements
A = np.array([range(1,5),('2','3','2','1')])
A.dtype


# In[17]:


B = np.array([range(1,5),(2.3,1.1)])
B.dtype


# In[18]:


# A.size--return the number of elements in the array
A = np.array([range(1,5),range(2,6)])
A.size


# In[19]:


# A.flat--return a one dimensional iterator over the array
A = np.array([range(1,4),(2,3,4),(1,0,1)])
B = A.flat
print(B)


# In[20]:


# A.itemsize--return the length of one array element in bytes
X = np.array([range(1,4),('0','0','1')])
X.itemsize


# In[21]:


Y = np.array([range(1,4),(0,0,1)])
Y.itemsize


# In[22]:


Z = np.array([1,3,5,7], dtype=np.float)
Z.itemsize


# # Slicing Numpy Array

# ## Slicing 1-Dimension NP Array

# In[23]:


y = np.arange(1,10)
print(y)
x = y[2:5]
print(x)


# In[24]:


y = np.array([1,5,6,7,8,5,6,7,0,1])
x = y[:8]
print(x)
z = y[:8:2]
print(z)


# In[25]:


y = np.array([1,5,6,7,8,5,6,7,0,1])
p = y[::]
print(p)
q = y[::2]
print(q)


# In[26]:


y = np.arange(1,10)
x = y[::-1]
print(x)
y = y[9:1:-1]
print(y)


# ## Slicing Multidimensional NP Array

# In[27]:


y = np.array([range(1,5),(2,3,4,0),range(2,6)])
A = y[0:2:]
print("A : ",A)
B = y[0:2,0:2]
print("B : ",B)
C = y[:,0:3]
print("C : ",C)


# In[28]:


# Extract Sub-Numpy Array by Boolean Selection
x = np.array([range(1,12)])
A = (x>4)
print(A)
B = (x>4)|(x<10)
print(x[A])
print(x[B])


# In[29]:


# Boolean operation on two NP Arrays
x = np.array([range(1,7)])**2
y = np.array([range(7,13)])
A = (x>y)   # here x and y must have same array size
B = (x==y)  # compare each pair (based on array index) of items in x and y
print('x is ',x)
print('A is ',A,',x[A] is ',x[A])
print('y is ',y)
print('B is ',B,',x[B] is ',x[B])


# # Fancy index on NP Array

# In[30]:


x = np.array([range(1,23)])
x.resize((6,4))
print(x)


# ## Several Typical Fancy Indexing

# In[31]:


# Extract a 1-dimension array Y from a 1-dimension array X
X = np.array(range(1,25))*100
Y = X[[1,3,0,12]]
print(Y)


# In[32]:


# Extract a 1-D array Y from a multidimensional NP array X
X = np.array(range(1,25))
X.resize((6,4))
print(X)
Y = X[[2,1,0],[0,3,1]]
print('Y is ',Y)


# In[33]:


x = np.array(range(1,17)).reshape((4,4))
print(x)
print(x[[1,2,0,3],:][[1]])
print(x[[1,2,0,3]][[1],:])
print(x[[0,1,2,3]][:,[1,2]])
print(x[[0,3,1,1]][[1,2]])


# # Reshaping/Transpose Numpy Array

# In[34]:


# Reshape Numpy array ravel and flatten
x = np.array([range(1,5), np.zeros(4), (1,2,1,2)])
print(x)
A = x.ravel()
B = x.flatten()
print('ravel : ',A)
print('flatten : ',B)
A[1] = 100
print(x)
B[1] = 200
print(x)


# In[35]:


# Reshape Numpy Array
x = np.array([range(1,5), np.zeros(4), (1,2,1,2)])
print('x is ',x)
A = x.transpose()
B = x.T
print(A)
print(B)
A[2,2] = 100
print(x)
B[2,2] = -9
print(x)


# In[36]:


# resize and reshape
x = np.array(range(1,13))
y = np.array(range(1,13))
z = x
A = x.reshape(3,4)
print('A',A)
y.resize((4,3))
print('y',y)

# change the shape
x.shape = (2,6)
print(x)
print(z)


# # Merging Numpy Arrays

# In[37]:


# Merging in vertical & horizontal way of 1-D
x = np.array([range(1,5)]).reshape(2,2)
y = np.array([range(3,7)]).reshape(2,2)
print(x)
print(y)

# vertical way
print(np.concatenate((x,y), axis=0))
print(np.vstack((x,y)))

# horizontal way
print(np.concatenate((x,y), axis=1))
print(np.hstack((x,y)))


# In[38]:


# Stacking two single dimension NP Arrays
x = np.array([range(1,5)])
y = np.array([range(3,7)])
print(np.row_stack((x,y)))
print(np.column_stack((x,y)))


# # Splitting Numpy Arrays

# In[39]:


z = np.array([range(1,5), range(3,7)])
print(z)

print(z.shape)

print(np.hsplit(z,4))

print(np.split(z,4, axis=1))

print(np.vsplit(z,2))

print(np.split(z,2, axis=0))


# # Data Analysis for numpy array
# ## Data processing method of NP array

# In[40]:


# .copy()--obtain a copy of NP array
x = np.array([[11,12,0],[5,1,8]])
print('x is ',x)
z = x.copy()
print('z is ',z)


# In[41]:


# .fill()--fill a NP array with a scalar value
x = np.array(range(1,10))
x.fill(20)
print(x)
a = np.empty(6)
print(a)
b = a.reshape(2,3)
b.fill(100)
print(b)


# In[42]:


# .tolist()--return a python list by converting a NP array
z = np.array([[11,12,0],[5,1,8]])
print(z)
A = z.tolist()
print(A,type(A[0]))
B = list(z)
print(B)
print(type(B[0]))


# In[43]:


# .itemset()--use a scalar to update item(at given position) of NP array
x = np.array([[1,2,3],[4,9,6],[0,1,0]])
print(x)
x.itemset(3,100)  # update the 4th value
print(x)
x.itemset((0,0),100)  # update the 1st value
print(x)


# In[44]:


# .sort()--sort an array in NP
a = np.array([2,1,3,0])
a.sort()
print(a)

x = np.array([[1,6,2],[4,0,2],[1,7,0]])
x.sort(axis=1)  # sort by column
print(x)

x.sort(axis=0)  # sort by row
print(x)


# In[45]:


# np.diff()--calculate the difference with different lags(order) b/w two consecutive elements in NP
A = np.array([1,2,3,4,5,7,9])
B = np.diff(A)
C = np.diff(B)
D = np.diff(A,n=2)
print(A)
print(B)
print(C)
print(D)


# In[46]:


# .nonzero()--return the indices of elements that are non-zero. The result is a 2D array
y = np.array([1,0,2,0,6,0])
z = y.nonzero()
print(z)

x = np.array([[1,6,2],[4,0,2],[1,7,0]])
print(x)
print(x.nonzero())


# In[47]:


# np.isnan()--return bollean NP array containing value True if the element is NaN; otherwise value is False
y = np.array([1,0,np.NaN,0,np.NaN,0,9])
z = np.isnan(y)  # get bollena NP array
print(z)

s = z.sum()  # total missing number
print(s)

print(np.mean(z))


# In[48]:


# .repeat()--return NP array by repeating elements of specified NP array
a = np.array([[1,3,2],[4,5,2],[0,7,0]])
print('a is ',a)

b = a.repeat(2,axis=None)
print('b is ',b)  # repeat 2 in 1 dimension array

c = a.repeat(2,axis=1)
print('c is ',c)  # repeat 2 over the column

x = np.array([[1,2],[3,4]])
y = np.repeat(x,2)
print('y is ',y)

z = np.repeat(x,[1,2],axis=0)
print('z is ',z)   #repeat [1,2] over the row


# In[49]:


# .choose()--use an index array to construct a new array from set of choices
y = [[9,1,7],[21,18,0],[8,1,13]]
print('y is ',y)

z = np.choose([1,2,0],y)
print('z is ',z)

x = [np.arange(1,5),np.arange(11,15),np.arange(21,25)]
print('x is ',x)

A = np.choose([1,0,2,1],x)
print('A is',A)


# In[50]:


# .take()--return an array based on index elements of other array
x = np.array(range(1,7))
print('x is ',x)
i = [0,3,4,2,0]
A = np.take(x,i)
print('A is ',A)


# In[51]:


# .argsort()--returns the indices that are ranking orders of NP array
z = np.array([55,11,92,10,12,70])
print(np.argsort(z))

B = z.reshape(2,3)
print('B is ',B)
print(np.argsort(B,axis=1))
print(np.argsort(B,axis=0))
print(np.argsort(B,axis=None))


# In[52]:


# .searchsorted()--find indices where elements of array should be inserted in to keep original order
x = np.array(range(1,7))*100
print('x is ',x)
print(np.searchsorted(x,230))
print(np.searchsorted(x,230,side='right'))
print(np.searchsorted(x,[-80,220,370,600]))


# In[53]:


# .where()--conditionally selected array elements
x = np.array([3,1,2,5,8])
y = np.zeros(5)
condition = np.array([False,False,True,False,True])
A = np.where(condition,x,y)
print('A is ',A)

B = np.where(x>2,1,0)
print('B is ',B)

C = np.where(x<2,0,x)
print('c is ',c)


# In[54]:


# .compress()--return selected slices of array over the axis (0 or 1)
x = np.array([[11,22],[4,11],[15,21]])
print('x is ',x)

A = np.compress([0,1,0],x,axis=0)
print('A is ',A)  # only choose the second row

B = np.compress([False,False,True],x,axis=0)
print('B is ',B)  # choose the last row

C = np.compress([False,True],x,axis=1)
print('C is ',C)  # only choose the second column


# # Sampling and Data Generation

# In[55]:


# np.random.rand(M,N)--generate a uniform distributed random M*N matrix, with each item's value b/w 0&1
x = np.random.rand(3,3)
print(x)


# In[56]:


# np.random.randn(M,N)--generate standard normal distributed random M*N matrix
x = np.random.randn(3,3)
print(x)


# In[57]:


# np.random.randint(low,high=None,size=None)
print(np.random.randint(1,8,size=6))
print(np.random.randint(1,9,size=(2,3)))
print(np.random.randint(5,size=(2,2)))  #lower bound is 0 and upperbound is 5-1=4

