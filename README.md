
# Handwritten Digit Recognition Model

This project implements a Handwritten Digit Recognition model using Convolutional Neural Networks (CNN), leveraging the power of TensorFlow, Keras, and Pandas for preprocessing and building the model. The model achieves an impressive accuracy of over 97% on the test dataset.

# Table of Contents
- [overview](#overview)
- [Dataset](#Dataset)
- [Model Architecture]()
- [Training](#Training)
- [Evaluation](#Evaluation)
- [Confusion Matrix]()
- [Results](#Results)
- [Dependencies](#Dependencies)
- [How to Run](#)

# Overview

The project is designed to recognize handwritten digits from the popular MNIST dataset, a large dataset of 60,000 training images and 10,000 test images of handwritten digits from 0 to 9. This model uses a CNN architecture combined with the Adam optimizer and sparse categorical crossentropy as the loss function to classify the images.

# Dataset

The MNIST dataset contains grayscale images of handwritten digits, each of size 28x28 pixels. For more information on the MNIST dataset, you can refer to the official page here.

You can download the dataset using TensorFlow with the following command:

```bash
 https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

```

# Model Architecture

The model is built using Keras Sequential API and consists of the following layers:

* Dense Layer 1: 100 units with ReLU activation
* Dense Layer 2: 10 units with Sigmoid activation

```bash
 model = keras.Sequential([
    keras.layers.Dense(100, activation='relu'),  # input layer
    keras.layers.Dense(10, activation='sigmoid')  # output layer
])

```

# Training

The model was trained using 5 epochs, and the data was flattened before being fed into the network.

```bash
model.fit(x_train_flatten, y_train, epochs=5)

```

# Evaluation

After training, the model achieved 97.41% accuracy on the test dataset.
```bash
model.evaluate(x_test_flatten, y_test)

```
# Output:
```bash
313/313 [==============================] - 1s 3ms/step - loss: 0.0802 - accuracy: 0.9741
[0.0802217647433281, 0.9740999937057495]

```

# Confusion Matrix

The confusion matrix provides a more detailed understanding of the model's performance, showing how well the model classifies each digit.

```bash
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred_labels)

```

# Results

* Test Loss: 0.0802
* Test Accuracy: 97.41%
The confusion matrix above shows that the model performs well on most of the digits, with only a few misclassifications, demonstrating the model's robustness in recognizing handwritten digits.

# How to Run

1. Clone the repository:
```bash
git clone https://github.com/shalupatidar22/HandWritten_Digit_Recognition.git

```

2. Navigate to the project directory
```bash
cd HandWritten_Digit_Recognition
```
3. Install the dependencies:
```bash
pip install -r requirements.txt

```

4. Run the project:
```bash
python train_model.py

```



