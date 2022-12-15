import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.array([-7.0,-4.0,-1.0,2.0,5.0,8.0,11.0,14.0])
Y = np.array([3.0,6.0,9.0,12.0,15.0,18.0,21.0,24.0])

plt.scatter(X,Y)
# plt.show()
print(Y==X+10)

X = tf.constant(X)
Y = tf.constant(Y)

print(X,Y)