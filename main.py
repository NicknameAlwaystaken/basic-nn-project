from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable
import math
import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def activate(self, value) -> float:
        pass

    @abstractmethod
    def derivative(self, value) -> float:
        pass


class ReluActivation(ActivationFunction):
    def activate(self, value):
        return max(0, value)

    def derivative(self, value):
        return 1 if value > 0 else 0


class SigmoidActivation(ActivationFunction):
    def activate(self, value):
        return 1 / (1 + math.exp(-value))

    def derivative(self, value):
        sig = self.activate(value)
        return sig * (1 - sig)


class Node:
    _id = 0

    def __init__(self, function: ActivationFunction):
        self.weights = []
        self.bias = 0.0
        self.activation_value = 0.0
        self.activation_function = function.activate
        self.derivative_function = function.derivative
        self.input_values = []
        self.net_input = 0.0
        self.gradient = 0.0

        self.id = Node._id
        Node._id += 1

    def __str__(self):
        return f"Node ID: {self.id}, Input Values: {self.input_values}, Activation Value: {self.activation_value} Input Values: {len(self.input_values)} Weights: {len(self.weights)}"


class Layer:
    def __init__(self, inputs: int, nodes: int, activation_function: Callable):
        self.inputs = inputs
        self.node_list: list[Node] = []
        self.deviation = self.gaussian_deviation(self.inputs)
        self.output = []

        self._create_nodes(nodes, activation_function)

    def _create_nodes(self, nodes, activation_function):
        for _ in range(nodes):
            node = Node(activation_function)
            limit = 1 / math.sqrt(self.inputs)  # Xavier initialization for sigmoid
            node.weights = np.random.uniform(-limit, limit, self.inputs).tolist()
            node.bias = 0.0

            self.node_list.append(node)

    def forward_propagate(self, input_list: list):
        self.output = []
        for node in self.node_list:
            node.net_input = 0
            node.input_values = input_list
            node.net_input = sum(w * x for w, x in zip(node.weights, input_list)) + node.bias
            node.activation_value = node.activation_function(node.net_input)

            self.output.append(node.activation_value)

    def gaussian_deviation(self, inputs):
        return 1 / math.sqrt(inputs)

    def nodes(self):
        return len(self.node_list)


class NeuralNetwork:
    def __init__(self, input_size, learning_rate):
        self.layers: list[Layer] = []
        self.learning_rate = learning_rate
        self.input_size = input_size

    def add_layer(self, nodes, activation_function):
        if not self.layers:
            inputs = self.input_size
        else:
            inputs = self.layers[-1].nodes()
        new_layer = Layer(inputs, nodes, activation_function)
        self.layers.append(new_layer)

    def predict(self, input_values: list):
        layer_output = input_values
        for layer in self.layers:
            layer.forward_propagate(layer_output)
            layer_output = layer.output

        return layer_output

    def back_propagate(self, targets: list):
        layers_backwards = self.layers[::-1]
        for i, layer in enumerate(layers_backwards):
            if i == 0:
                for index, node in enumerate(layer.node_list):
                    gradient = node.activation_value - targets[index]
                    node.gradient = gradient

                    for key in range(len(node.weights)):
                        node.weights[key] -= self.learning_rate * gradient * node.input_values[key]

                    node.bias -= self.learning_rate * gradient
            else:
                next_layer = layers_backwards[i - 1]

                for index, node in enumerate(layer.node_list):
                    error = 0

                    for _, next_node in enumerate(next_layer.node_list):
                        weight_from_current_to_next = next_node.weights[index]
                        error += weight_from_current_to_next * next_node.gradient

                    gradient = error * node.derivative_function(node.net_input)
                    node.gradient = gradient

                    for key in range(len(node.weights)):
                        node.weights[key] -= self.learning_rate * gradient * node.input_values[key]

                    node.bias -= self.learning_rate * gradient


def mse_loss(predictions: list, targets: list):
    if len(predictions) != len(targets) or not predictions or not targets:
        raise ValueError("predictions and/or targets aren't valid")
    else:
        errors = []
        for i in range(len(predictions)):
            difference = predictions[i] - targets[i]
            errors.append(difference**2)

        return sum(errors) / len(errors)


def binary_cross_entropy_loss(predictions: list, targets: list):
    epsilon = 1e-15  # To prevent log(0)
    total_loss = 0
    for prediction, target in zip(predictions, targets):
        prediction = max(min(prediction, 1 - epsilon), epsilon)
        total_loss += -(target[0] * math.log(prediction) + (1 - target[0]) * math.log(1 - prediction))
    return total_loss / len(predictions)


if __name__ == "__main__":
    network = NeuralNetwork(2, learning_rate=0.1)
    network.add_layer(2, SigmoidActivation())
    network.add_layer(1, SigmoidActivation())
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = [[0], [1], [1], [0]]
    epochs = 10000

    for epoch in range(epochs):
        total_loss = 0
        for i, input_data in enumerate(inputs):
            target = targets[i]
            predictions = network.predict(input_data)
            loss = binary_cross_entropy_loss(predictions, [target])
            total_loss += loss
            network.back_propagate(target)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(inputs)}")

    print("\nTesting the trained network:")
    for input_data in inputs:
        output = network.predict(input_data)
        print(f"Input: {input_data}, Output: {output}")

    network = NeuralNetwork(2, learning_rate=0.1)
    network.add_layer(2, SigmoidActivation())
    network.add_layer(1, SigmoidActivation())
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = [[0], [1], [1], [0]]
    epochs = 10000

    for epoch in range(epochs):
        total_loss = 0
        for i, input_data in enumerate(inputs):
            target = targets[i]
            predictions = network.predict(input_data)
            loss = binary_cross_entropy_loss(predictions, [target])
            total_loss += loss
            network.back_propagate(target)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(inputs)}")

    print("\nTesting the trained network:")
    for input_data in inputs:
        output = network.predict(input_data)
        print(f"Input: {input_data}, Output: {output}")
