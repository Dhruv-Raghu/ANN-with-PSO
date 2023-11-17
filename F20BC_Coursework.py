import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(0)
   
class Activation:
    def __init__(self, activation_type):
       self.activation_type = activation_type

    def evaluate(self, x):
       if self.activation_type == 'logistic':
           return 1 / (1 + np.exp(-x))
       elif self.activation_type == 'ReLU':
           return np.maximum(0, x)
       elif self.activation_type == 'tanh':
           return np.tanh(x)
    
    def derivate(self, x):
       if self.activation_type == 'logistic':
           return self.evaluate(x) * (1 - self.evaluate(x))
       elif self.activation_type == 'ReLU':
           return 1 * (x > 0)
       elif self.activation_type == 'tanh':
           return 1 - np.power(self.evaluate(x), 2)
       
class Neuron:
    def __init__(self, weights, bias, activation):
        self.weights = weights
        self.bias = bias
        self.activation = activation
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        total = np.dot(self.weights, inputs) + self.bias
        output = self.activation.evaluate(total)
        self.output = output
        return output
    
    def backward(self, d):
        d = d * self.activation.derivate(self.output)
        self.bias = self.bias - d
        self.weights = self.weights - np.dot(self.inputs, d)
        return d
    
class Layer:
    def __init__(self, num_neurons, num_inputs, activation):
        self.num_neurons = num_neurons
        self.weights = np.random.randn(num_neurons, num_inputs) * np.sqrt(2 / (num_inputs + num_neurons))
        self.bias = np.random.randn(num_neurons, 1)
        self.activation = activation
        self.layer = self.createLayer()
    
    def createLayer(self):
        layer = []
        for x in range(self.num_neurons):
            layer.append(Neuron(self.weights[x], self.bias[x], self.activation))
        return layer
    
    def forward(self, inputs):
        # print(inputs)
        # print(self.weights)
        # print(self.bias)

        outputs = []
        for neuron in self.layer:
            outputs.append(neuron.forward(inputs))
        return np.array(outputs).flatten()
    
    def backward(self, d):
        for neuron in self.layer:
            d = neuron.backward(d)
        return d
    
    def size(self):
        return self.num_neurons
    
class NeuralNetwork:
    def __init__(self, X, y):
        self.layers = []
        self.X = X
        self.y = y
        self.classes = np.unique(y)

    def appendLayer(self, layer):
        self.layers.append(layer)

    def createLayer(self, num_neurons, activation):
        if len(self.layers) == 0:
            num_inputs = self.X.shape[1]
        else:
            num_inputs = self.layers[-1].size()

        num_outputs = num_neurons

        # Initialize weights and bias in the Layer class
        layer = Layer(num_outputs, num_inputs, activation)
        self.layers.append(layer)

    def forward(self, row_data):
        output = row_data
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    
    def predict(self, X_test):
        predictions = []
        for x in range(X_test.shape[0]):
            output = self.forward(X_test[x])
            # print(output)
            prediction = self.classes[np.argmax(output)]
            predictions.append(prediction)
        return np.array(predictions)
    
    
def main():
    # 2 rows of data to test forward function
    X_trial = np.array([[3.6216,8.6661,-2.8073,-0.44699], [-1.3971,3.3191,-1.3927,-1.9948]])
    y_trial = np.array([0, 1])
    # print(input_data.shape)

    # binary classification dataset
    data_csv = pd.read_csv('/Users/dhruv/Documents/Y4S1/F20BC/Coursework/data.csv')
    X = data_csv.iloc[:, :-1].to_numpy()
    y = data_csv.iloc[:, -1].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create network
    network = NeuralNetwork(X_trial, y_trial)
    network.createLayer(3, Activation('ReLU'))
    network.createLayer(2, Activation('logistic'))

    # test forward function
    # forward_test = network.forward(X_trial[1])
    # print(forward_test)
    # predict_test = network.predict(X_trial)
    # print(predict_test)

    network2 = NeuralNetwork(X_train, y_train)
    network2.createLayer(3, Activation('ReLU'))
    network2.createLayer(2, Activation('logistic'))

    # # test forward function
    # forward_test = network.forward(X_trial[1])
    # print(forward_test)
    predict_test = network2.predict(X_test)
    print(predict_test)


if __name__ == "__main__":
    main()