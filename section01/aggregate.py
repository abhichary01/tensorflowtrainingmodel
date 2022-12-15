import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

D = tf.constant([-7,-10])

print(tf.reduce_min(D))
print(tf.reduce_max(D))
print(tf.reduce_mean(D))
print(tf.reduce_sum(D))

E = tf.constant(np.random.randint(0,100,size = 50))
print(E)

# Variance

print(tfp.stats.variance(E))
print(tf.math.reduce_std(tf.cast(E,dtype=tf.float16)))
print(tf.math.reduce_variance(tf.cast(E,dtype=tf.float16)))

#Finding postional maximum and minimum

tf.random.set_seed(42)
F = tf.random.uniform(shape=[50])
print(F)
print(tf.argmax(F))
print(F[tf.argmax(F)])
print(tf.reduce_max(F))
print(F[tf.argmax(F)] == tf.reduce_max(F))

print(tf.argmin(F))
print(F[tf.argmin(F)])

G = tf.constant(tf.random.uniform(shape=[50]),shape=(1,1,1,1,50))
print(G)
print(G.shape)
G_Squeezed = tf.squeeze(G)
print(tf.squeeze(G))
print(G_Squeezed.shape)

# One hot encoding tensors

some_list = [0,1,2,3]

print(tf.one_hot(some_list,depth=4))

H = tf.range(1,10)
print(tf.square(H))
print(tf.sqrt(tf.cast(H, dtype=tf.float32)))

print(tf.math.log(tf.cast(H, dtype=tf.float32)))

J=tf.constant(np.array([3.,7.,10.]))
print(np.array(J))

print(tf.config.list_physical_devices("GPU"))