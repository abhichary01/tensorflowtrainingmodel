import tensorflow as tf

tensor = tf.constant([[10,7],[3,4]])

print(tensor+10)

print(tf.multiply(tensor,10))



print(tf.matmul(tensor,tensor))
print(tensor*tensor)

X = tf.constant([[1,2],[3,4],[5,6]])
XX = tf.constant([[10,11],[7,8],[9,10]])
Y = tf.constant([[7,8,9,24,6],[9,10,11,25,4]])

print(X*XX)

print(tf.matmul(X,Y))#m*n = n*m  n should be same

print(tf.tensordot(X,Y,axes=1))

print(tf.tensordot(tf.transpose(X),XX,axes=1))