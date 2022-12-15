import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

# X = np.array([-7.0,-4.0,-1.0,2.0,5.0,8.0,11.0,14.0])
# Y = np.array([3.0,6.0,9.0,12.0,15.0,18.0,21.0,24.0])

# X = tf.cast(tf.constant(X),dtype=tf.float32)
# Y = tf.cast(tf.constant(Y),dtype=tf.float32)
# Model

# tf.random.set_seed(42)

# Same as above with dense layers
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(50, activation="relu"),
#     tf.keras.layers.Dense(1)])

# model.add(tf.keras.layers.Dense(1))

# Compiling the model

# model.compile(loss="mae",
# optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
# metrics=["mae"])

# model.fit(tf.expand_dims(X, axis=-1), Y, epochs=5)
# model.fit(tf.expand_dims(X, axis=-1), Y, epochs=100)

# ypred = model.predict([17.0])
# print(ypred,tf.shape(ypred))

X = tf.range(-100,100,4)
Y = X+10

print(len(X))

# plt.scatter(X,Y)
# plt.show()

X_train = X[:40]
Y_train = Y[:40]
X_test = X[40:]
Y_test=X[40:]

model = tf.keras.Sequential([
    # tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(10, input_shape=[1],name="input_layer"),
    tf.keras.layers.Dense(1,name="output_layer")],
    name="model_1")

model.compile(loss=tf.keras.losses.mae,
optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
metrics=["mae"])

model.fit(tf.expand_dims(X, axis=-1), Y, epochs=100, verbose=1)

model.summary()

# plt.figure(figsize=(10,7))
# plt.scatter(X_train,Y_train,c="b",label="Training data")
# plt.scatter(X_test, Y_test,c="g",label="Testing data")
# plt.legend()
# plt.show()

plot_model(model=model, show_shapes=True)

y_pred = model.predict(X_test)
print(y_pred)

def plot_predictions(train_data=X_train,
train_lables=Y_train,
test_data=X_test,
test_lables=Y_test,
predictions=y_pred):

    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_lables, c="b",label="Training data")
    plt.scatter(test_data, test_lables, c="g",label="Testing data")
    plt.scatter(test_data, predictions, c="r",label="predictions")
    plt.legend()

    # plt.show()

# plot_predictions()

print(model.evaluate(X_test,Y_test))
tf.squeeze(y_pred) #removes one dimension (10,1) to (10,)

mae = tf.metrics.mean_absolute_error(y_true=Y_test,y_pred=tf.squeeze(y_pred))
print(mae)

mse = tf.metrics.mean_squared_error(y_true=Y_test,y_pred=tf.squeeze(y_pred))
print(mse)

def maeerror(y_true,y_pred):
    return tf.metrics.mean_absolute_error(y_true=y_true,y_pred=tf.squeeze(y_pred))

def mseerror(y_true,y_pred):
    return tf.metrics.mean_squared_error(y_true=y_true,y_pred=tf.squeeze(y_pred))
