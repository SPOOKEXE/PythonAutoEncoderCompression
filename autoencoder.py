import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from sklearn.model_selection import train_test_split

# https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
# https://www.tensorflow.org/tutorials/generative/autoencoder

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class NumberVectorGenerator:

	@staticmethod
	def generate( shape : tuple ) -> np.ndarray:
		return np.random.randint(low=0, high=255, size=shape)

class SimpleAutoEncoder(keras.Model):

	def __init__(self, shape):
		super(SimpleAutoEncoder, self).__init__()

		self.shape = shape

		self.encoder = keras.Sequential([
			keras.layers.Flatten(input_shape=shape, name='flatten'),
			keras.layers.Dense(50*50*3, activation='relu', name='dense_1'),
			keras.layers.Dense(30*30*3, activation='relu', name='dense_2'),
			keras.layers.Dense(10*10*3, activation='relu', name='dense_3'),
			keras.layers.Dense(3*3*3, activation='relu', name='dense_4'),
		], name="encoder")

		self.decoder = keras.Sequential([
			keras.layers.Dense(3*3*3, input_shape=(3*3*3,), activation='relu', name='dense_1'),
			keras.layers.Dense(10*10*3, activation='relu', name='dense_2'),
			keras.layers.Dense(30*30*3, activation='relu', name='dense_3'),
			keras.layers.Dense(50*50*3, activation='sigmoid', name='dense_4'),
			keras.layers.Reshape(target_shape=shape, name='reshape'),
		], name="decoder")

	def call( self, x ) -> tuple:
		# print( np.shape(x) )
		encoded = self.encoder( x )
		# print( np.shape(encoded) )
		decoded = self.decoder(encoded)
		# print( np.shape(decoded) )
		return decoded#, encoded

autoencoder = SimpleAutoEncoder(shape=(50*50, 3))
autoencoder.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())

print(autoencoder.encoder.summary(), autoencoder.decoder.summary())

data = NumberVectorGenerator.generate( (10000, 50*50, 3) )
print( np.shape(data) )

x_train, x_test = train_test_split( data, test_size=0.2, shuffle=True )
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

history = autoencoder.fit(
	x_train,
	x_train,
	epochs=4,
	shuffle=True,
	batch_size=64,
	validation_data=(x_test, x_test),
	use_multiprocessing=True
)

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
	# display original
	ax = plt.subplot(2, n, i + 1)
	plt.imshow(x_test[i])
	plt.title("original")
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	# display reconstruction
	ax = plt.subplot(2, n, i + 1 + n)
	plt.imshow(decoded_imgs[i])
	plt.title("reconstructed")
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()