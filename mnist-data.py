# Ploting mnist instances
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Loading the data, if not present then Downloading
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))

plt.show()