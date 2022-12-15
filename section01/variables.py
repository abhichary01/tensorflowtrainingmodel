import tensorflow as tf
import numpy as np
changable_tensor = tf.Variable([10,7])
unchangable_tensor = tf.constant([10,7])
changable_tensor, unchangable_tensor

changable_tensor[0].assign(7)
print(changable_tensor)

random_1 = tf.random.Generator.from_seed(42)
random_1 = random_1.normal(shape=(3,2))
random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.normal(shape=(3,2))

random_1,random_2, 
print(random_1 == random_2)

not_shuffled = tf.constant([[10,7],[8,6],[9,2],[8,22]])
print(not_shuffled.ndim)

tf.random.set_seed(42) #  with same seed same output at different places global level
print("y",tf.random.shuffle(not_shuffled, seed=42))#operational level

numpy_A = np.arange(1,25,dtype=np.int32)

print(numpy_A)

A = tf.constant(numpy_A, shape=(2,3,4))
B = tf.constant(numpy_A)

print(A,"\n",B)

rank4_tensor = tf.zeros(shape=(2,3,4,5))
print(rank4_tensor)
print('s',rank4_tensor[0])

print("**************************")

print(rank4_tensor.shape,'\n',rank4_tensor.ndim,'\n',tf.size(rank4_tensor))

print(rank4_tensor[:2,:2,:2,:2])

rank2_tensor = tf.constant([[1,4],[8,9]])
rank3_tensor = rank2_tensor[...,tf.newaxis]
print('rank2 and 3',rank2_tensor,'\n',rank3_tensor)

# alternative to add axis

print(tf.expand_dims(rank2_tensor,axis=-1))

print(tf.expand_dims(rank2_tensor,axis=0))

print(rank2_tensor)