import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils.vis_utils import plot_model

from model2 import maeerror, mseerror, plot_predictions


tf.random.set_seed(42)

X = tf.range(-100,100,4)
Y = X+10

# X = tf.cast(tf.constant(X),dtype=tf.float32)
# Y = tf.cast(tf.constant(Y),dtype=tf.float32)

X_train = X[:40]
Y_train = Y[:40]
X_test = X[40:]
Y_test=X[40:]


model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

model_1.compile(loss=tf.keras.losses.mae,
optimizer=tf.keras.optimizers.SGD(),metrics=["mae"])

model_1.fit(tf.expand_dims(X_train, axis=-1), Y_train, epochs=100)

y_preds_1 = model_1.predict(X_test)
plot_predictions(predictions=y_preds_1)

mae_1 = maeerror(Y_test, y_preds_1)
mse_1 = mseerror(Y_test,y_preds_1)

print(mae_1,mse_1)

# model 2  with 2 or more dense layers

model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

model_2.compile(loss=tf.keras.losses.mae,
optimizer = tf.keras.optimizers.SGD(),metrics=["mse"])

model_2.fit(tf.expand_dims(X_train, axis=-1), Y_train, epochs=100)

y_preds_2 = model_2.predict(X_test)
plot_predictions(predictions=y_preds_2)

mae_2 = maeerror(Y_test,y_preds_2)
mse_2=mseerror(Y_test,y_preds_2)

print(mae_2,mse_2)

model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

model_3.compile(loss=tf.keras.losses.mae,
optimizer=tf.keras.optimizers.SGD(),metrics=["mae"])

model_3.fit(tf.expand_dims(X_train, axis=-1), Y_train, epochs=100)

y_preds_3 = model_3.predict(X_test)
plot_predictions(predictions=y_preds_3)

mae_3 = maeerror(Y_test,y_preds_3)
mse_3=mseerror(Y_test,y_preds_3)

print(mae_3,mse_3)

model_results = [["model_1", mae_1.numpy(), mse_1.numpy()],
["model_2",mae_2.numpy(),mse_2.numpy()],
["model_3",mae_3.numpy(),mse_3.numpy()]]

all_results = pd.DataFrame(model_results, columns=["model","mae","mse"])

print("All results",all_results)

# model_1.save("model_01 save from mode.py of tensorflow") 

# model_1.save("savedwith_HDF5_format.h5")