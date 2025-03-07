# Neural Network

## Introduction
This repository contains a simple neural network built from scratch using NumPy. The network is designed for experimentation and learning, with a focus on understanding fundamental deep learning concepts. Currently, it processes and learns from test inputs stored within the neural network file.

## Features
- **Fully implemented in NumPy**: No external deep learning libraries like TensorFlow or PyTorch.
- **Three main components**:
  - `neuron.py`: Implements individual neurons with weights, biases, and activation functions.
  - `layers.py`: Manages multiple neurons within a layer.
  - `network.py`: Combines multiple layers to form the complete network.
- **Activation Functions**: Uses both **Sigmoid** and **ReLU** for learning.
- **Basic Learning Capability**: The model currently processes and learns from predefined test inputs.
- **Weight Storage and Retrieval**: Allows saving and loading trained weights for reuse.

## Project Structure
```
├── neuron.py       # Implementation of individual neurons
├── layers.py       # Layer management containing multiple neurons
├── network.py      # Neural network combining multiple layers
├── README.md       # Project documentation
```

## Installation
Clone the repository and install dependencies:
```sh
git clone https://github.com/parsaferdosi/neural-network.git
cd neural-network
pip install numpy
```


## Usage
### Running the Network
To run the network and test its learning capabilities, execute:
```sh
python network.py
```
This will process the test inputs and train the model.

## Future Plans
- Adapt the neural network for OCR (Optical Character Recognition) applications
- Implement image preprocessing techniques for better text recognition
- Train the model on real-world datasets instead of test inputs
- Explore different architectures for improved performance

## Contributing
Feel free to fork this repository and contribute! Open an issue or submit a pull request if you have any ideas or improvements.

and thanks chatGPT for helping me make this readme file