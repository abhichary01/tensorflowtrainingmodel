from pydoc import plain
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

 
n_samples = 1000

X,y = make_circles(n_samples,noise=0.03,
random_state=42)

circles = pd.DataFrame({"X0":X[:,0],"X1":X[:,1],"label":y})


print(circles)
print(X.shape,y.shape)
plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.RdYlBu)
# plt.show()

tf.random.set_seed(42)

model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1)
])

model_1.compile(loss=tf.keras.losses.binary_crossentropy,
optimizer=tf.keras.optimizers.SGD(),metrics=["accuracy"])

# model_1.fit(tf.expand_dims(X, axis=-1), y, epochs=15)

model_1.fit(tf.expand_dims(X, axis=-1), y, epochs=100,verbose=0)
model_1.evaluate(X,y)

print(circles["label"].value_counts())

tf.random.set_seed(42)
model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

model_2.compile(loss=tf.keras.losses.BinaryCrossentropy(),
optimizer=tf.keras.optimizers.Adam(),metrics=["accuracy"])

model_2.fit(tf.expand_dims(X, axis=-1), y, epochs=100,verbose=0)
model_2.evaluate(X,y)


def plot_decision_boundary(model, X, y):
    X_min, X_max = X[:,0].min()-0.1,X[:,0].max()+0.1
    Y_min, Y_max = X[:,1].min()-0.1,X[:,1].max()+0.1
    XX, YY = np.meshgrid(np.linspace(X_min,X_max,100),
    (np.linspace(Y_min,Y_max,100)))

    X_in =np.c_[XX.ravel(),YY.ravel()]
    Y_pred=model.predict(X_in)

    if len(Y_pred[0])>1:
        print("doing multi class classification")

        Y_pred = np.argmax(Y_pred,axis=1).reshape(XX.shape)

    else:
        print("doing binary classification")
        Y_pred = np.round(Y_pred).reshape(XX.shape)

    plt.contourf(XX,YY,Y_pred,cmap=plt.cm.RdYlBu,alpha=0.7)
    plt.scatter(X[:,0],X[:,1],c=y,s=40,cmap=plt.cm.RdYlBu)
    plt.xlim(XX.min(),XX.max())
    plt.xlim(YY.min(),YY.max())
    # plt.show()

plot_decision_boundary(model=model_2,X=X,y=y)

tf.random.set_seed(42)
X_regression = tf.range(0,1000,5)
Y_regression = tf.range(100,1100,5)

X_reg_train = X_regression[:150]
X_reg_test = X_regression[150:]
Y_reg_train = Y_regression[:150]
Y_reg_test = Y_regression[150:]
model_2.fit(tf.expand_dims(X_reg_train, axis=-1), Y_reg_train, epochs=100,verbose=0)

model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

model_3.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.Adam(),
metrics=["mae"])

model_3.fit(tf.expand_dims(X_reg_train, axis=-1), Y_reg_train, epochs=100,verbose=0)

y_reg_preds = model_3.predict(X_reg_test)
plt.figure(figsize=(10,7))
plt.scatter(X_reg_train,Y_reg_train,c="b",label="Training data")
plt.scatter(X_reg_test,Y_reg_test,c="g",label="Test data")
plt.scatter(X_reg_test, y_reg_preds,c="r",label="predictions")
plt.legend()
# plt.show()

tf.random.set_seed(42)

model_4 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
])

model_4.compile(loss="binary_crossentropy",
optimizer=tf.keras.optimizers.Adam(lr=0.001),
metrics=["accuracy"])

history = model_4.fit(X,y,epochs=100)

plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.RdYlBu)
# plt.show()

plot_decision_boundary(model=model_4,X=X,y=y)

model_5 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation=tf.keras.activations.relu)
])

model_5.compile(loss=tf.keras.losses.BinaryCrossentropy(),
optimizer=tf.keras.optimizers.Adam(lr=0.001),
metrics=["accuracy"])

model_5.fit(X,y,epochs=100)

print("@@@@@@@")
tf.random.set_seed(42)
model_6  = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model_6.compile(loss="binary_crossentropy",
optimizer=tf.keras.optimizers.Adam(lr=0.001),
metrics=["accuracy"])

history06 = model_6.fit(X, y, epochs=100,verbose=0)
print("////////////////")
model_6.evaluate(X,y)
plot_decision_boundary(model=model_6,X=X,y=y)
# plt.show()