# Predictive Coding for Image Classification (MNIST)

This project implements a minimal predictive coding neural network with lateral connections for image classification on the MNIST dataset. The model is inspired by neuroscience and demonstrates how prediction errors can guide both state and weight updates, offering an alternative to standard backpropagation.

## Features

- Predictive coding model with three hidden layers and lateral (within-layer) connections
- Manual, biologically inspired state and weight updates (no autograd for state updates)
- Trains and evaluates on the MNIST handwritten digit dataset
- Plots training/validation accuracy and loss over epochs

## Project Structure

```
.
├── pcmnist.py        # Main code: model, training, evaluation, plotting
├── pcmnist.ipynb     # (Optional) Jupyter notebook version
├── requirements.txt  # Python dependencies
└── data/             # MNIST data (downloaded automatically)
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YonatanBest/Predictive-Coding-for-Image-Classfication
cd Predictive-Coding-for-Image-Classification
```

### 2. Install dependencies

It's recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Run the code

```bash
python pcmnist.py
```

This will:
- Download and normalize the MNIST dataset
- Train the predictive coding model
- Print training/validation loss and accuracy per epoch
- Display plots of accuracy and loss

## Results

- Achieves ~89-90% validation accuracy after 10 epochs
- Demonstrates the effectiveness of predictive coding with lateral connections for image classification

## Files

- **pcmnist.py**: Main script with model, training, and evaluation code
- **Report for Predictive Coding for Image Classfication without Activation Functions.pdf**: [2 page summary of theory, implementation, and results](https://drive.google.com/file/d/1n55V7dMj97DaOskTbXfUcWJMLtzLwINs/view?usp=drive_link)
- **Report for Predictive Coding for Image Classfication with Activation Functions.pdf**: [2 page summary of theory, implementation, and results](https://drive.google.com/file/d/1nBK85VXoFmZ8QqJQg03Fr-ck2v4gW5sa/view?usp=drive_link)

- **requirements.txt**: List of required Python packages

## License

This project is for educational and research purposes.
