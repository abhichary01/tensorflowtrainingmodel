import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Read in the insurance dataset
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

# print(insurance["smoker"],insurance["age"])

k = pd.get_dummies(insurance)

X = k.drop("charges",axis=1)
Y = k["charges"]

print(X)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
print(len(X),len(X_train),len(X_test))

tf.random.set_seed(42)


insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

insurance_model.compile(loss=tf.keras.losses.mae,
optimizer=tf.keras.optimizers.SGD(),
metrics=["mae"])

s = insurance_model.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100)

d = insurance_model.evaluate(X_test,y_test)
print(d)

print(y_train.median(),y_train.mean())

pd.DataFrame(s.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")
plt.show()

# make two other models and compare later