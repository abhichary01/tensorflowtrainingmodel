import tensorflow as tf
print(tf.__version__)

scalar =tf.constant(7)
print(scalar)
print(scalar.ndim)

vector = tf.constant([10,10])
print(vector)
print(vector.ndim)

matrix = tf.constant([[11,7],[7,10]])
print(matrix)
print(matrix.ndim)

another_matrix = tf.constant([[10.,7.],[3.,2.],[8.,9.]],dtype=tf.float16)
print(another_matrix)
print(another_matrix.ndim)

tensor = tf.constant([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]],[[13,14,15],[16,17,18]]])
print(tensor)
print(tensor.ndim)