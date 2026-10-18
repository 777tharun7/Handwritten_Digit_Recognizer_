<div align="center">

# Handwritten Digit Recognizer

**Convolutional neural network trained on the Keras MNIST dataset**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![MNIST](https://img.shields.io/badge/Dataset-MNIST-39D98A?style=flat-square)](https://keras.io/api/datasets/mnist/)

</div>

## Overview

A computer-vision notebook that loads MNIST through `tensorflow.keras.datasets`, prepares the digit images and labels, trains a CNN, evaluates it on the MNIST test split, and saves a Keras model artifact.

## Implementation

- Notebook: [mnist.ipynb](mnist.ipynb)
- Saved model: [mnist_digit_recognizer.h5](mnist_digit_recognizer.h5)
- Model layers in the notebook include `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense`.
- The notebook trains for five epochs in its recorded run.

## Run

```bash
python -m pip install numpy matplotlib tensorflow pillow
```

Open `mnist.ipynb` in Jupyter or Google Colab and execute the cells in order. Keras downloads the MNIST dataset when it is first requested, so network access may be needed for initial setup.

## Evaluation

The saved notebook output reports **97.10% test accuracy** on the Keras MNIST test split. This is the recorded notebook result and has not been independently rerun during this repository audit.

For a stronger evaluation report, keep the test split untouched during model selection, add a confusion matrix and per-class metrics, and show examples of incorrect predictions. Test performance on MNIST does not guarantee performance on photographs or handwritten images from other sources.

## Repository

[Open source repository](https://github.com/777tharun7/Handwritten_Digit_Recognizer_)
