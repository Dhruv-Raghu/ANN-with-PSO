import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# np.random.seed(1911)

# Create an activation object and pass it to the neuron object
class Activation:
    def __init__(self, activation_type):
       self.activation_type = activation_type

    def evaluate(self, x):
        if self.activation_type == 'logistic':
            if x > 0:
                return 1 / (1 + np.exp(-x))
            else:
                return np.exp(x) / (1 + np.exp(x))
            
        elif self.activation_type == 'ReLU':
            return np.maximum(0, x)
        
        elif self.activation_type == 'leakyReLU':
            return np.where(x > 0, x, 0.01 * x)
        
        elif self.activation_type == 'tanh':
            return np.tanh(x)

# The neuron object calculates the weighted sum of all of its inputs, adds the bias, and passes the result through the activation function      
class Neuron:
    def __init__(self, weights, bias, activation):
        self.weights = weights
        self.bias = bias
        self.activation = activation    # activation object
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        total = np.dot(self.weights, inputs) + self.bias    # dot product of weights and neuron inputs + bias
        output = self.activation.evaluate(total)            # pass the neuron output through the activation function
        self.output = output
        return output

# The layer object creates a layer of neurons with the specified number of neurons, inputs, and activation function
class Layer:
    def __init__(self, num_neurons, num_inputs, activation):
        self.num_neurons = num_neurons
        self.weights = np.random.randn(num_neurons, num_inputs) * np.sqrt(2 / (num_inputs + num_neurons))   # randomize initial weights
        self.bias = np.random.randn(num_neurons, 1)     # randomize initial bias
        self.activation = activation
        self.layer = self.createLayer()
    
    # returns an array of neurons with the specified parameters
    def createLayer(self):
        layer = []
        for x in range(self.num_neurons):
            layer.append(Neuron(self.weights[x], self.bias[x], self.activation))
        return layer
    
    # send the inputs through each neuron in the layer and return the outputs in an array
    def forward(self, inputs):
        outputs = []
        for neuron in self.layer:
            outputs.append(neuron.forward(inputs))
        return np.array(outputs).flatten()
    
    # once the weights/ bias are updated by the optimization algorithm, update the weights/ bias in each neuron as well
    def updateWeights(self):
        iteration = 0
        for neuron in self.layer:
            neuron.weights = self.weights[iteration]
            neuron.bias = self.bias[iteration]
            iteration += 1
    
    def size(self):
        return self.num_neurons

# The neural network object creates a neural network with the specified number of layers, neurons, and activation functions
class NeuralNetwork:
    def __init__(self, numHiddenLayers=2, numNeurons=[8, 8], activation=['ReLU', 'ReLU']):
        self.numHiddenLayers = numHiddenLayers
        self.numNeurons = numNeurons
        self.activation = activation

        self.layers = []
        self.X = None
        self.y = None
        self.classes = None

    # assigns the input data and output labels to the network
    # creates the specified number of layers with the specified number of neurons and activation functions according to the data
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.fitComplete = True

        for x in range(self.numHiddenLayers):
            if x == 0:
                num_inputs = self.X.shape[1]
            else:
                num_inputs = self.layers[-1].size()

            num_outputs = self.numNeurons[x]

            # Initialize weights and bias in the Layer class
            layer = Layer(num_outputs, num_inputs, Activation(self.activation[x]))
            self.layers.append(layer)

        # Create output layer
        num_inputs = self.layers[-1].size()
        num_outputs = len(self.classes)
        layer = Layer(num_outputs, num_inputs, Activation('logistic'))
        self.layers.append(layer)

    # returns an array of all the weights and bias in the network
    def get_params(self):
        params = []
        for layer in self.layers:
            params.append(layer.weights.flatten())
            params.append(layer.bias.flatten())
        return np.concatenate(params)
    
    # function that takes one row of data and feeds it through the entire network
    # returns an array of all the raw neuron otuputs from the output layer
    def forward(self, row_data):
        output = row_data
        for layer in self.layers:
            output = layer.forward(output)
        return output  
    
    # runs the data row by row on the network, and returns an array of the predicted class for each row of data
    def predict(self, X_test):
        predictions = []
        for x in range(X_test.shape[0]):
            output = self.forward(X_test[x])
            # print(output)
            prediction = self.classes[np.argmax(output)]
            predictions.append(prediction)
        return np.array(predictions)
    
    # trains the network using PSO
    def train_with_PSO(self, num_particles, num_iterations, inertia, c1, c2, num_informants=6, dynamicParams=False):
        # Initialize PSO with test parameters
        pso = PSO(
            num_particles=num_particles,
            max_iter=num_iterations,
            inertia=inertia,
            c1=c1,  # cognitive_param
            c2=c2,  # social_param
            num_informants=num_informants
        )

        # Optimize the neural network using PSO
        if dynamicParams:
            optimized_network = pso.optimize(self, dynamicParams=True)
        else:
            optimized_network = pso.optimize(self)

        # replace the layers in the network with the optimized layers
        self.layers = optimized_network.layers


# The PSO object optimizes the neural network by updating the weights and bias in each layer
class PSO:
    def __init__(self, num_particles, max_iter, inertia, c1, c2, num_informants=5):
        self.num_particles = num_particles
        self.particles = []
        self.max_iter = max_iter
        self.alpha = inertia      # determines how much of the particle's current velocity should be retained
        self.beta = c1     # determines how much of the particle's best position should be retained
        self.gamma = c2       # determines how much of the informants's best position should be used
        self.num_informants = num_informants

    def optimize(self, network, dynamicParams=False):
        self.createParticles(network)

        best_particleFitness = 0
        best_network = None

        for x in range(self.max_iter):
            print("Iteration: ", x)
            print("Best fitness: ", best_particleFitness)

            for particle in self.particles:

                if dynamicParams:
                    # dynamically change inertia, cognitive, social, and global parameters with iteration
                    if x > 0.25 * self.max_iter:
                        self.alpha = max(0.8, self.alpha - 0.1)  # Reduce exploration
                        self.beta = max(0.8, self.beta - 0.1)
                        self.gamma = min(0.6, self.gamma + 0.1)
                    elif x > 0.5 * self.max_iter:
                        self.alpha = max(0.5, self.alpha - 0.1)
                        self.beta = max(0.6, self.beta - 0.1)
                        self.gamma = min(1.2, self.gamma + 0.1)
                    elif x > 0.75 * self.max_iter:
                        self.alpha = max(0.4, self.alpha - 0.1)
                        self.beta = max(0.4, self.beta - 0.1)
                        self.gamma = min(1.5, self.gamma + 0.1)


                if x > 0:
                    self.updateInformantBest(particle)
                    particle.updateVelocity(self.alpha, self.beta, self.gamma)
                    particle.updatePosition()
                particle.evaluateFitness()

                if particle.fitness > best_particleFitness:
                    best_particleFitness = copy.deepcopy(particle.fitness)
                    best_network = copy.deepcopy(particle.position)

            global_best = best_network

        optimized_network = best_network
        return optimized_network

    def createParticles(self, network):
        for x in range(self.num_particles):
            self.particles.append(Particle(randomizeNetwork(network)))

    # find k closest neighbours to the particle based on weights and biases
    def findNeighbours(self, particle):
        k = self.num_informants

        # Extract flattened parameters for the current particle
        particle_params = particle.position.get_params()

        # Calculate Euclidean distances between the particle and all other particles
        distances = []
        for other_particle in self.particles:
            if other_particle != particle:
                other_params = other_particle.position.get_params()
                distance = np.linalg.norm(particle_params - other_params)
                distances.append(distance)

        # Get indices of k closest neighbors
        closest_indices = np.argsort(distances)[:k]

        # Retrieve the k closest neighbors
        neighbours = [self.particles[i] for i in closest_indices]

        return neighbours
    
    def updateInformantBest(self, particle):
        neighbours = self.findNeighbours(particle)  # Find neighbours of the particle using numInformants
    
        best_neighbour = max(neighbours, key=lambda x: x.fitness) # Find the neighbour with the best fitness

        # update the particle's informant best if the neighbour has a better fitness
        if best_neighbour.fitness > particle.informant_best_fitness:
            particle.informant_best = copy.deepcopy(best_neighbour)
            particle.informant_best_fitness = copy.deepcopy(best_neighbour.fitness)
    
# The particle object of a PSO algorithm
# the position of the particle is a neural network object with the weights and bias of each layer
# the weights and bias of the particle are updated by the PSO algorithm
class Particle:
    def __init__(self, network, bound=5.0):
        self.position = copy.deepcopy(network)
        self.best_position = network
        self.informant_best = None
        self.fitness = 0
        self.best_fitness = 0
        self.informant_best_fitness = 0
        self.velocities = []
        self.initializeVelocity()
        self.bound = bound

        # self.num_informants = np.random.randint(3, 7) # will implement informants later

    # For each layer, it generates a random array (weights_velocity) with the same shape as the weights of that layer.
    def initializeVelocity(self):
        # For each layer, it generates 2 random arrays (weights_velocity, bias_velocity) with the same shape as the weights and bias of that layer.
        # multiply by 0.01 to make sure the initial velocity is small to promote balanced exploration of the search space
        for layer in self.position.layers:
            weights_velocity = np.random.randn(*layer.weights.shape) * 0.01
            bias_velocity = np.random.randn(*layer.bias.shape) * 0.01
            self.velocities.append((weights_velocity, bias_velocity))

    def updateVelocity(self, alpha, beta, gamma):
        # alpha -> inertia_param      # determines how much of the particle's current velocity should be retained
        # beta -> cognitive_param     # determines how much of the particle's best position should be retained
        # gamma -> social_param       # determines how much of the informants's best position should be retained

        # Update velocities element-wise
        for i, (weights_vel, bias_vel) in enumerate(self.velocities):
            weights_update = alpha * weights_vel + beta * np.random.uniform(0, 1) * (self.best_position.layers[i].weights - self.position.layers[i].weights) + gamma * np.random.uniform(0,1) * (self.informant_best.position.layers[i].weights - self.position.layers[i].weights)
            bias_update = alpha * bias_vel + beta * np.random.uniform(0, 1) * (self.best_position.layers[i].bias - self.position.layers[i].bias) + gamma * np.random.uniform(0,1) * (self.best_position.layers[i].bias - self.informant_best.position.layers[i].bias)

            # Update velocities
            self.velocities[i] = (weights_update, bias_update)

    def updatePosition(self):
        for i, layer in enumerate(self.position.layers):
            # Update weights 
            layer.weights = layer.weights + self.velocities[i][0]
            # and apply reflective bounding
            layer.weights = self.reflectiveBound(layer.weights, -self.bound, self.bound)  

            # Update biases 
            layer.bias = layer.bias + self.velocities[i][1]
            # and apply reflective bounding
            layer.bias = self.reflectiveBound(layer.bias, -self.bound, self.bound)  

            layer.updateWeights()   # update the neurons with the new weights
    
    def reflectiveBound(self, value, lower_bound, upper_bound):
        # Reflective bounding for an array of values
        for i in range(len(value)):
            value[i] = np.where(value[i] < lower_bound, 2 * lower_bound - value[i], value[i])
            value[i] = np.where(value[i] > upper_bound, 2 * upper_bound - value[i], value[i])
        return value
            
    # evaluate the fitness of the particle by running the train data through the network and comparing the output to the labels
    def evaluateFitness(self):
        y_pred = self.position.predict(self.position.X)
        self.fitness = get_accuracy(self.position.y, y_pred)
        # if the fitness of the particle is better than the best fitness, update the best position and best fitness of the particle
        if self.fitness > self.best_fitness:
            self.best_position = copy.deepcopy(self.position)
            self.best_fitness = copy.deepcopy(self.fitness)


def get_accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

# rnadomize the weights and bias of the provided network
def randomizeNetwork(network):
    for x in range(len(network.layers)):
        network.layers[x].weights = np.random.randn(network.layers[x].num_neurons, network.layers[x].weights.shape[1]) * np.sqrt(2 / (network.layers[x].weights.shape[1] + network.layers[x].num_neurons))
        network.layers[x].bias = np.random.randn(network.layers[x].num_neurons, 1)
        network.layers[x].updateWeights()

    return network
    
def main_ANN_Test():
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
    network = NeuralNetwork()
    network.fit(X_trial, y_trial)
    network.createLayer(3, Activation('ReLU'))
    # network.createLayer(2, Activation('logistic'))

    # test forward function
    # forward_test = network.forward(X_trial[1])
    # print(forward_test)
    # predict_test = network.predict(X_trial)
    # print(predict_test)

    network2 = NeuralNetwork()
    network2.fit(X_train, y_train)
    network2.createLayer(3, Activation('ReLU'))
    network2.createLayer(2, Activation('logistic'))

    # # test forward function
    # forward_test = network.forward(X_trial[1])
    # print(forward_test)
    predict_test = network2.predict(X_test)
    print(predict_test)
    print(get_accuracy(y_test, predict_test))

    network2 = randomizeNetwork(network2)
    predict_test = network2.predict(X_test)
    print(predict_test)
    print(get_accuracy(y_test, predict_test))

    # print(network.get_params())

def main_PSO_Test():
    # # Load data and split into training and testing sets
    # data_csv = pd.read_csv('/Users/dhruv/Documents/Y4S1/F20BC/Coursework/data.csv')
    # X = data_csv.iloc[:, :-1].to_numpy()
    # y = data_csv.iloc[:, -1].to_numpy()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # test on iris dataset
    iris = load_iris()
    iris_X = iris.data
    iris_y = iris.target
    print(np.unique(iris_y)) # [0 1 2]

    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # iris_X = scaler.fit_transform(iris_X)

    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

    # take an average of 15 runs
    accuracyList = []
    for i in range(15):
        # Create a neural network
        network = NeuralNetwork(numHiddenLayers=1, numNeurons=[8], activation=['ReLU'])
        network.fit(X_train, y_train)

        # Initialize PSO with test parameters
        num_particles = 30
        max_iter = 100
        inertia_param = 1.0
        cognitive_param = 1.0
        social_param = 1.0

        # Optimize the neural network using PSO
        network.train_with_PSO(num_particles, max_iter, inertia_param, cognitive_param, social_param, num_informants=6)

        # Evaluate the performance of the optimized network on test data
        predictions = network.predict(X_test)
        accuracy = get_accuracy(y_test, predictions)
        print("Accuracy on test data ", i ,":", accuracy) 
        accuracyList.append(accuracy)
    
    print("Average accuracy: ", np.mean(accuracyList))
       

def main_MLP_Test():
    from sklearn.neural_network import MLPClassifier

    # Load data and split into training and testing sets
    iris = load_iris()
    iris_X = iris.data
    iris_y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.2)

    accuracyList = []
    # take an average of 10 runs
    for i in range(10):
        # Create a neural network
        clf = MLPClassifier(hidden_layer_sizes=(1, 14), activation='tanh', solver='sgd', learning_rate_init=0.01, max_iter=50)
        clf.fit(X_train, y_train)

        # Evaluate the performance of the optimized network on test data
        predictions = clf.predict(X_test)
        accuracy = get_accuracy(y_test, predictions)
        print("Accuracy on test data ",i,":", accuracy)
        accuracyList.append(accuracy)
    
    print("Average accuracy: ", np.mean(accuracyList))

def visualizeCourseworkData():
    # visualize the data to see if its linearly separable
    import matplotlib.pyplot as plt
    data_csv = pd.read_csv('/Users/dhruv/Documents/Y4S1/F20BC/Coursework/data.csv')
    X = data_csv.iloc[:, :-1].to_numpy()
    y = data_csv.iloc[:, -1].to_numpy()

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #import PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

def visualizeIrisData():
    # visualize the data to see if its linearly separable
    import matplotlib.pyplot as plt
    iris = load_iris()
    iris_X = iris.data
    iris_y = iris.target

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    iris_X = scaler.fit_transform(iris_X)

    #import PCA
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)
    # iris_X = pca.fit_transform(iris_X)

    plt.scatter(iris_X[:, 0], iris_X[:, 1], c=iris_y)
    plt.show()


if __name__ == "__main__":
    # visualizeCourseworkData()
    # visualizeIrisData()
    # main_ANN_Test()
    main_PSO_Test()
    # main_MLP_Test()