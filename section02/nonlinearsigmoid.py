from pydoc import plain
from random import random
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from classification import X, plot_decision_boundary,y
from sklearn.metrics import confusion_matrix
import itertools

A = tf.cast(tf.range(-10,10),tf.float32)

def sigmoid(x):
    return 1/(1+tf.exp(-x))

s = sigmoid(A)
print(s)

plt.plot(s)
# plt.show()

def relu(x):
    return tf.maximum(0,x)

plt.plot(relu(A))
# plt.show()

X_train, y_train = X[:800],y[:800]
X_test, y_test = X[800:], y[800:]

X_train.shape, X_test.shape, y_train.shape, y_test.shape

tf.random.set_seed(42)

model_8 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model_8.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.Adam(lr=0.01),
metrics=["accuracy"])

history = model_8.fit(X_train,y_train,epochs=25)
model_8.evaluate(X_test,y_test)
print("Check",model_8.evaluate(X_test,y_test))

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_8,X=X_train,y=y_train)
plt.subplot(1,2,2)
plt.title("test")
plot_decision_boundary(model_8,X=X_test,y=y_test)
print("DDDD++++++++")
# plt.show()

pd.DataFrame(history.history).plot()
plt.title("Model_8 loss curves")
# plt.show()

tf.random.set_seed(42)

model_9 = tf.keras.Sequential([
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model_9.compile(loss="binary_crossentropy",
optimizer="Adam",metrics=["accuracy"])

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-4*10**(epoch/20)
)

history_9 = model_9.fit(X_train,y_train,epochs=100,
callbacks=[lr_scheduler])

pd.DataFrame(history_9.history).plot(figsize=(10,7),xlabel="epochs")
# plt.show()

# ideal learning rate plot

lrs = 1e-4*(10**(tf.range(100)/20))
plt.figure(figsize=(10,7))
plt.semilogx(lrs, history_9.history["loss"])
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning rate vs loss")
# plt.show()

tf.random.set_seed(42)

model_10 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model_10.compile(loss="binary_crossentropy",
optimizer=tf.keras.optimizers.Adam(lr=0.02),metrics=["accuracy"])

history_10 = model_10.fit(X_train,y_train,epochs=20)

print(model_10.evaluate(X_test, y_test))

plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_10,X=X_train,y=y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_10,X=X_test,y=y_test)

# plt.show()

loss, accuracy = model_10.evaluate(X_test, y_test)
print(f"Model loss:{loss}")
print(f"Model accuracy:{accuracy}")


y_preds = model_10.predict(X_test)
tf.round(y_preds)[:10]
print("ssss",confusion_matrix(y_test,tf.round(y_preds)))

def make_confusion_matrix(y_true,y_pred,classes=None,figsize=(10,10),text_size=15):

    cm = confusion_matrix(y_true,y_pred)
    cm_norm = cm.astype("float")/cm.sum(axis=1)[:,np.newaxis]
    n_classes = cm.shape[0]

    fig, ax =plt.subplots(figsize=figsize)
    cax = ax.matshow(cm,cmap=plt.cm.Blues)
    fig.colorbar(cax)

    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    ax.set(title = "Confusion matrix",
    xlabel = "predicted label",
    ylabel="True Label",
    xticks=np.arange(n_classes),
    yticks=np.arange(n_classes),
    xticklabels=labels,
    yticklabels=labels
    )
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    ax.title.set_size(text_size)

    threshold=(cm.max()+cm.min())/2.

    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i, f"{cm[i,j]} ({cm_norm[i,j]*100:.1f}%)",
        horizontalalignment="center",color="white" if cm[i,j]>threshold else "black",size=text_size)


# plt.show()