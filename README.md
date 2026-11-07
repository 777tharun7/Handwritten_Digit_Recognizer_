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

The saved notebook records **97.10% validation accuracy** at epoch 5. The notebook first reduces MNIST to 10,000 training examples and 2,000 test examples, then passes that same 2,000-example test subset as `validation_data` during training. It subsequently evaluates on that same subset. Therefore, 97.10% is a notebook-reported score on a subset already used for validation/model monitoring; it is **not an untouched independent test result** and has not been independently rerun during this audit.

For a defensible final test score, reserve a test set that is never passed to `model.fit`, use a separate validation split for model selection, and report the sample counts, random seed, confusion matrix, and per-class precision/recall/F1. MNIST performance does not guarantee performance on photographs or handwriting from other sources.

## Repository

[Open source repository](https://github.com/777tharun7/Handwritten_Digit_Recognizer_)
