import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import time

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize images

# Build CNN model
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
start_time = time.time()
cnn_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
end_time = time.time()

# Evaluate performance
test_loss, test_accuracy = cnn_model.evaluate(x_test, y_test)
training_time = end_time - start_time

print(f"CNN Test Accuracy: {test_accuracy}")
print(f"CNN Training Time: {training_time} seconds")


cnn_model.save("cnn_mnist_model.h5")