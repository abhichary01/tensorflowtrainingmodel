import tensorflow as tf
from model01 import model_2
from model01 import X_test


# loaded_model = tf.keras.models.load_model("/model_01 save from mode.py of tensorflow")
# loaded_model.summary()

# model01.model_2.summary()

loadedh5_model=tf.keras.models.load_model("/savedwith_HDF5_format.h5")

loadedh5_model.predict(X_test)

model_2.predict(X_test) == loadedh5_model.predict(X_test)