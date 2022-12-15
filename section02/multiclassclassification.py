import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
from pydoc import plain
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from nonlinearsigmoid import make_confusion_matrix
import itertools

(train_data, train_labels), (test_data,test_labels)= fashion_mnist.load_data()

print(f"Training sample:\n {train_data[0]}\n")
print(f"Training label:\n {train_labels[0]}\n")

print(train_data[0].shape,train_labels[0].shape)


print(train_labels[7])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.imshow(train_data[9],cmap=plt.cm.binary)
# plt.title(class_names[train_labels[9]])

import random
plt.figure(figsize=(7,7))
for i in range(4):
    ax=plt.subplot(2,2,i+1)
    random_index = random.choice(range(len(train_data)))
    plt.imshow(train_data[random_index],cmap=plt.cm.binary)
    plt.title(class_names[train_labels[random_index]])
    plt.axis(False)


tf.random.set_seed(42)

model_11 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(10,activation=tf.keras.activations.softmax),
])

model_11.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
optimizer=tf.keras.optimizers.Adam(),
metrics=["accuracy"])

non_norm_history = model_11.fit(train_data,
tf.one_hot(train_labels,depth=10),
epochs=10,validation_data=(test_data,tf.one_hot(test_labels,depth=10)))

print(model_11.summary())

train_data_norm = train_data/255.0
test_data_norm = test_data/255.0

print(train_data_norm.min(),train_data_norm.max())

tf.random.set_seed(42)

model_12 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(10,activation=tf.keras.activations.softmax)
])

model_12.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
optimizer=tf.keras.optimizers.Adam(),
metrics=["accuracy"])

norm_history = model_12.fit(train_data_norm,
train_labels,epochs=10,
validation_data=(test_data_norm,test_labels))

pd.DataFrame(non_norm_history.history).plot(title="non normalized data")
pd.DataFrame(norm_history.history).plot(title=" normalized data")


tf.random.set_seed(42)

model_13 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(10,activation="softmax")
])

model_13.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
optimizer=tf.keras.optimizers.Adam(),
metrics=["accuracy"])

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-3*10**(epoch/20))


find_lr_history = model_13.fit(train_data_norm,
train_labels,epochs=10,
validation_data=(test_data_norm,test_labels),
callbacks=[lr_scheduler])

# lrs = 1e-3*(10**(tf.range(40)/20))
# plt.semilogx(lrs,find_lr_history.history["loss"])
# plt.xlabel("Learning rate")
# plt.ylabel("loss")
# plt.title("finding ideal learning rate")


model_14 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(4,activation="relu"),
    tf.keras.layers.Dense(10,activation="softmax")
])

model_14.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
optimizer=tf.keras.optimizers.Adam(lr=0.001),
metrics=["accuracy"])

history_14 = model_14.fit(train_data_norm,
train_labels,epochs=10,
validation_data=(test_data_norm,test_labels))

y_probs = model_14.predict(test_data_norm)

y_preds = y_probs.argmax(axis=1)
y_preds[:10]

make_confusion_matrix(y_true=test_labels,
y_pred = y_preds,classes=class_names,
figsize=(15,15),text_size=10)

def plot_random_image(model,images,true_labels,classes):
    i = random.randint(0,len(images))
    target_image=images[i]
    pred_probs = model.predict(target_image.reshape(1,28,28))
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[true_labels[i]]

    plt.imshow(target_image,cmap=plt.cm.binary)

    if pred_label == true_label:
        color="green"
    else:
        color = "red"
    plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(pred_label,
    100*tf.reduce_max(pred_probs),
    true_label),color=color)

plot_random_image(model=model_14,images=test_data_norm,
true_labels=test_labels,classes=class_names
)

weights,biases = model_14.layers[1].get_weights()
print(weights,biases)

# plt.show()