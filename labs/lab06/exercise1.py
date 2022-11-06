import numpy as np
import matplotlib .pyplot as plt

x = np.array ([[0 , 0],
[0, 1],
[1, 0],
[1, 1]])

y = np.array ([0,
1,
1,
0])

print(np.shape(x))
print(np.shape(y))

print(len(x)) # e q u i v a l e n t t o np . s h a pe ( x ) [ 0 ]
print(len(y)) # e q u i v a l e n t t o np . s h a pe ( y ) [ 0 ]

print(x[1])
print(y[1])

print(x[: ,0])

print(x[-1,:])
print(y[-1])

x = 2*x-1 # M u l t i p l y a l l e n t r i e s i n x by 2 and t h e n s u b t r a c t 1
print(x)

x_row_sum = x[0 ,:]+x[1 ,:]+x[2 ,:]+x[3 ,:]
x_row_sum = np.sum(x,axis =0)

x = np.zeros( shape =(50 ,3) ) #C r e a t e s a 2−dim numpy a r r ay o f z e r o s ,
#e s s e n t i a l l y a 50 x3 m a t r i x o f z e r o s
x = np.ones( shape =(45 ,2) ) #C r e a t e s a 2−dim numpy a r r ay o f ones ,
#e s s e n t i a l l y a 45 x2 m a t r i x o f one s
x = np.random.rand (12 ,4) #C r e a t s a 2−dim numpy a r r ay o f random
# v a l u e s , e s s e n t i a l l y a 12 x4 m a t r i x o f
# random v a l u e s
x = np.arange(start =2, stop =8, step =0.5)
#C r e a t e s a 1−dim numpy a r r ay o f v a l u e s
#s t a r t i n g from 2 , i n c r em e n t i n g by 0 . 5
# and g o i n g up t o ( b u t n o t i n c l u d i n g ) 8 .
x = np.linspace(start =0, stop =50, num =13)
# C r e a t e s a 1−dim numpy a r r ay o f 13 v a l u e s
# e q u a l l y s p a c e d be twee n 0 and 50 ( b u t n o t
# i n c l u d i n g 5 0 )

x = np.array ([2001 , 2002 , 2003 , 2004 , 2005])
y = np.array ([1 ,4 ,9 ,0.5 ,25])
plt.scatter(x,y,marker='x',color='black')
plt.show()

plt.plot(x,y,color='blue')

plt.scatter(x,y,marker='x',color='black', label="Scatter")
plt.plot(x,y,color='blue',label="Line")
plt.xlabel('Years')
plt.ylabel('Sales (in K$)')
plt.legend()
plt.show()

x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0,
              1,
              1,
              0])

x = 2*x-1

z = x[:, 0]*x[:, 1]

x = np.array([2001, 2002, 2003, 2004, 2005])
y = np.array ([1 ,4 ,9 ,0.5 ,25])

plt.scatter(x,y,marker='x',color='black')
y = np.array([0, 4, 8, 12, 16])
plt.plot(x, y, color='red',label="Polynomial degree 1")
y = np.array([1, 4, 9, 16, 25])
plt.plot(x, y, color='green',label="Polynomial degree 2")
plt.legend()
plt.show()







