import math
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import time

from sklearn.metrics import classification_report, confusion_matrix

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
    def __init__(self, numHiddenLayers=1, numNeurons=[4], activation=['ReLU']):
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
    
    
    def loss(self):
        y_one_hot = np.eye(len(self.classes))[self.y]

        outputs =[]
        for i in range(self.X.shape[0]):
            output = np.array(self.forward(self.X[i]))
            outputs.append(output)

        outputs = np.array(outputs)

        # Clip predicted values to avoid log(0) issues
        epsilon = 1e-15
        outputs = np.clip(outputs, epsilon, 1 - epsilon)

        # Calculate cross-entropy loss
        loss = - np.sum(y_one_hot * np.log(outputs) + (1 - y_one_hot) * np.log(1 - outputs))

        # Normalize the loss by the number of samples
        loss /= len(y_one_hot)
            
        return loss
    
    # trains the network using PSO
    def train_with_PSO(self, num_particles, num_iterations, inertia, c1, c2, c3, num_informants=6, dynamicParams=False,):
        # Initialize PSO with test parameters
        pso = PSO(
            num_particles=num_particles,
            max_iter=num_iterations,
            inertia=inertia,
            c1=c1,  # cognitive_param
            c2=c2,  # social_param
            c3=c3,  # global_param
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
    def __init__(self, num_particles, max_iter, inertia, c1, c2, c3, num_informants=5):
        self.num_particles = num_particles
        self.particles = []
        self.max_iter = max_iter
        self.alpha = inertia      # determines how much of the particle's current velocity should be retained
        self.beta = c1     # determines how much of the particle's best position should be retained
        self.gamma = c2       # determines how much of the informants's best position should be used
        self.delta = c3     # determines how much of the global best position should be used
        self.num_informants = num_informants

        self.pso_best_fitness = math.inf
        self.pso_best_network = None

    def optimize(self, network, dynamicParams=False, showFitnessUpdate = False):
        self.createParticles(network)

        global_best_network = None
        global_best_fitness = math.inf

        if showFitnessUpdate:
            best_fitness_list = []

        print("Optimizing the network...")

        for x in range(self.max_iter):
            # print("Iteration: ", x)
            # print("Best fitness: ", global_best_fitness)

            if x == 0:
                print("0% complete")
            elif x == int(0.25*self.max_iter):
                print("25% complete")
            elif x == int(0.5*self.max_iter):
                print("50% complete")
            elif x == int(0.75*self.max_iter):
                print("75% complete")

            for particle in self.particles:

                if dynamicParams:

                    # Linearly Decreasing Inertia Weight
                    start_inertia = 0.9
                    end_inertia = 0.4
                    self.alpha = (start_inertia - end_inertia) * (self.max_iter - x) / self.max_iter + end_inertia

                    # if x <= 0.25 * self.max_iter:   # first quarter of iterations
                    #     # focus on exploring the search space
                    #     self.alpha = 0.9
                    #     self.beta = 2.0
                    #     self.gamma = 0.9
                    #     # self.num_informants = 3
                    # elif x > 0.25 * self.max_iter and x <= 0.5 * self.max_iter:
                    #     # balance exploration and exploitation
                    #     self.alpha = min(0.8, self.alpha - 0.1)
                    #     self.beta = min(1.5, self.beta - 0.1)
                    #     self.gamma = max(1.2, self.gamma + 0.1)
                    #     # self.num_informants = 4
                    # elif x > 0.5 * self.max_iter and x <= 0.75 * self.max_iter:
                    #     # focus on exploiting the search space
                    #     self.alpha = min(0.7, self.alpha - 0.1)
                    #     self.beta = min(1.2, self.beta - 0.1)
                    #     self.gamma = max(1.5, self.gamma + 0.1)
                    #     # self.num_informants = 6
                    # elif x > 0.75 * self.max_iter:
                    #     # focus on exploiting the search space
                    #     self.alpha = min(0.4, self.alpha - 0.1)
                    #     self.beta = min(0.9, self.beta - 0.1)
                    #     self.gamma = max(2.0, self.gamma + 0.1)
                    #     # self.num_informants = 10


                if x > 0:
                    self.updateInformantBest(particle)
                    particle.updateVelocity(self.alpha, self.beta, self.gamma, self.delta)
                    particle.updatePosition()
                particle.evaluateFitness_loss()

                if particle.fitness < global_best_fitness:
                    # global_best = copy.deepcopy(particle)
                    global_best_network = copy.deepcopy(particle.position)
                    global_best_fitness = copy.deepcopy(particle.fitness)
                    

                particle.global_best_network = global_best_network
                particle.global_best_fitness = global_best_fitness

            if showFitnessUpdate:
                best_fitness_list.append(global_best_fitness)
        # uncomment to plot the fitness of the best particle over time
        # if showFitnessUpdate:
        #     plt.title("PSO Global Best Fitness over time")
        #     plt.xlabel("Iteration")
        #     plt.ylabel("Global Best Fitness")

        #     # plot iteration on x-axis and fitness on y-axis
        #     best_fitness_list = np.array(best_fitness_list)
        #     plt.plot(best_fitness_list[:, 0], best_fitness_list[:, 1])

        #     plt.show()

        print("Optimization complete!")
        print("Best fitness: ", global_best_fitness)
        optimized_network = global_best_network
        
        if showFitnessUpdate:
            return optimized_network, best_fitness_list
        
        else:
            return optimized_network

    def createParticles(self, network):
        for x in range(self.num_particles):
            self.particles.append(Particle(randomizeNetwork(network)))
    
    # find k closest neighbours to the particle based on fitness
    def findNeighbours_fitness(self, particle):
        k = self.num_informants

        # Calculate Euclidean distances between the particle and all other particles
        distances = []
        for other_particle in self.particles:
            if other_particle != particle:
                distance = np.linalg.norm(particle.fitness - other_particle.fitness)
                distances.append(distance)

        # Get indices of k closest neighbors
        closest_indices = np.argsort(distances)[:k]

        # Retrieve the k closest neighbors
        neighbours = [self.particles[i] for i in closest_indices]

        return neighbours
    
    def updateInformantBest(self, particle):
        neighbours = self.findNeighbours_fitness(particle)  # Find neighbours of the particle using numInformants
    
        best_neighbour = max(neighbours, key=lambda x: x.fitness) # Find the neighbour with the best fitness

        # update the particle's informant best if the neighbour has a better fitness
        if best_neighbour.fitness < particle.informant_best_fitness:
            particle.informant_best_network = copy.deepcopy(best_neighbour.position)
            particle.informant_best_fitness = copy.deepcopy(best_neighbour.fitness)


# The particle object of a PSO algorithm
# the position of the particle is a neural network object with the weights and bias of each layer
# the weights and bias of the particle are updated by the PSO algorithm
class Particle:
    def __init__(self, network, bound=5.0):
        self.position = copy.deepcopy(network)
        self.best_position = network
        self.informant_best_network = network
        self.global_best_network = network
        self.fitness = math.inf
        self.best_fitness = math.inf
        self.informant_best_fitness = math.inf
        self.global_best_fitness = math.inf
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

    def updateVelocity(self, alpha, beta, gamma, delta):
        # alpha -> inertia_param      # determines how much of the particle's current velocity should be retained
        # beta -> cognitive_param     # determines how much of the particle's best position should be retained
        # gamma -> social_param       # determines how much of the informants's best position should be retained

        # Update velocities element-wise
        for i, (weights_vel, bias_vel) in enumerate(self.velocities):
            weights_update = alpha * weights_vel + beta * np.random.uniform(0, 1) * (self.best_position.layers[i].weights - self.position.layers[i].weights) + gamma * np.random.uniform(0,1) * (self.informant_best_network.layers[i].weights - self.position.layers[i].weights) + delta * np.random.uniform(0,1) * (self.global_best_network.layers[i].weights - self.position.layers[i].weights)
            bias_update = alpha * bias_vel + beta * np.random.uniform(0, 1) * (self.best_position.layers[i].bias - self.position.layers[i].bias) + gamma * np.random.uniform(0,1) * (self.informant_best_network.layers[i].bias - self.position.layers[i].bias) + delta * np.random.uniform(0,1) * (self.global_best_network.layers[i].bias - self.position.layers[i].bias)

            # Update velocities
            self.velocities[i] = (weights_update, bias_update)

    def updatePosition(self):
        for i, layer in enumerate(self.position.layers):
            # Update weights 
            layer.weights = layer.weights + self.velocities[i][0]
            layer.weights = self.reflectiveBound(layer.weights, -self.bound, self.bound)

            # Update biases 
            layer.bias = layer.bias + self.velocities[i][1]
            layer.bias = self.reflectiveBound(layer.bias, -self.bound, self.bound)

            layer.updateWeights()   # update the neurons with the new weights
    
    def reflectiveBound(self, values, lower_bound, upper_bound):
        # Reflective bounding for an array of values
        for i in range(len(values)):
            values[i] = np.where(values[i] < lower_bound, 2 * lower_bound - values[i], values[i])
            values[i] = np.where(values[i] > upper_bound, 2 * upper_bound - values[i], values[i])
        return values
    
    # deprecated     
    # evaluate the fitness of the particle by running the train data through the network and comparing the output to the labels
    def evaluateFitness(self):
        y_pred = self.position.predict(self.position.X)
        self.fitness = get_accuracy(self.position.y, y_pred)
        # if the fitness of the particle is better than the best fitness, update the best position and best fitness of the particle
        if self.fitness > self.best_fitness:
            self.best_position = copy.deepcopy(self.position)
            self.best_fitness = copy.deepcopy(self.fitness)

    # evaluate the fitness by calculating the loss of the network
    def evaluateFitness_loss(self):
        self.fitness = self.position.loss()
        # if the fitness of the particle is lower than the best fitness, update the best position and best fitness of the particle
        if self.fitness < self.best_fitness:
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
    # Load data and split into training and testing sets
    # data_csv = pd.read_csv('/Users/dhruv/Documents/Y4S1/F20BC/Coursework/data.csv')
    # X = data_csv.iloc[:, :-1].to_numpy()
    # y = data_csv.iloc[:, -1].to_numpy()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # test on iris dataset
    iris = load_iris()
    iris_X = iris.data
    iris_y = iris.target
    print(np.unique(iris_y)) # [0 1 2]

    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

    # take an average of 20 runs
    accuracyList = []
    for i in range(20):
        # Create a neural network
        network = NeuralNetwork(numHiddenLayers=1, numNeurons=[5], activation=['ReLU'])
        network.fit(X_train, y_train)

        # Initialize PSO with test parameters
        num_particles = 30
        max_iter = 100
        inertia_param = 0.5
        cognitive_param = 1.0
        social_param = 1.2
        global_param = 2.0

        # Optimize the neural network using PSO
        network.train_with_PSO(num_particles, max_iter, inertia_param, cognitive_param, social_param, global_param, num_informants=4)

        # Evaluate the performance of the optimized network on test data
        predictions = network.predict(X_test)
        accuracy = get_accuracy(y_test, predictions)
        print("Accuracy on test data ", i ,":", accuracy) 
        accuracyList.append(accuracy)
    
    print("Average accuracy: ", np.mean(accuracyList))
    
    # plot the accuracy of each run
    plt.plot(accuracyList)

def main_plot():
    # test on iris dataset
    iris = load_iris()
    iris_X = iris.data
    iris_y = iris.target
    print(np.unique(iris_y)) # [0 1 2]

    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)

    # Create a neural network
    network = NeuralNetwork(numHiddenLayers=1, numNeurons=[5], activation=['ReLU'])
    network.fit(X_train, y_train)
    
    pso = PSO(num_particles=30, max_iter=100, inertia=0.5, c1=1.0, c2=1.2, c3=2.0, num_informants=4)
    pso.createParticles(network)

    pso.optimize(network, showFitnessUpdate=True)

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

def coefficientTesting():
    # Load data and split into training and testing sets
    data_csv = pd.read_csv('data.csv')
    X = data_csv.iloc[:, :-1].to_numpy()
    y = data_csv.iloc[:, -1].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    inertia = 0.5
    num_particles = 25
    num_informants = 4
    num_iterations = 25

    c1_params = [0.8, 2.4, 0.8, 1.5, 1.3, 1.0]
    c2_params = [0.8, 0.8, 2.4, 1.5, 1.3, 1.0]
    c3_params = [2.4, 0.8, 0.8, 1.0, 1.4, 2.0]

    inertia_params = [0.2, 0.5, 0.8, 1.0]

    swarm_sizes = [15, 25, 25, 25, 50]
    iterations = [75, 50, 25, 15, 15]

    informants = [2, 6, 10]

    num_runs = 10    # for how many runs should the average be taken
    
    # Create a dataframe that stores the accuracy of the network on the test data for each run
    accuracy_df = pd.DataFrame(columns=range(num_runs), index=range(len(inertia_params)))
    # Create a datafram that stores the runtime of the network for each run
    runtime_df = pd.DataFrame(columns=range(num_runs), index=range(len(inertia_params)))

    for i in range(len(informants)):
        # optimal coefficients
        c1 = c1_params[5]
        c2 = c2_params[5]
        c3 = c3_params[5]

        # optimal inertia
        inertia = inertia_params[1]

        # optimal swarm size and number of iterations
        num_particles = swarm_sizes[2]
        num_iterations = iterations[2]

        num_informants = informants[i]

        # for coefficent testing
        # if i == 0:
        #     print("\nMore Emphasis on Global Exploration")
        # if i == 1:
        #     print("\nMore Emphasis on Personal Knowledge")
        # if i == 2:
        #     print("\nMore Emphasis on Swarm Knowledge")
        # if i == 3:
        #     print("\nEmphasis on Personal and Swarm Knowledge")
        # if i == 4:
        #     print("\nEqual Importance to all Parameters")
        # if i == 5:
        #     print("\nBalanced Exploration and Exploitation")

        # Create a dataframe that stores the best fitness of the PSO algorithm of each epoch for each run
        fitness_df = pd.DataFrame(columns=range(num_iterations), index=range(num_runs))

        # take an average of num_runs
        accuracyList = []
        runtimeList = []
        for j in range(num_runs):
            print("\nTest ", i, " Run ", j)

            # Create a neural network
            network = NeuralNetwork(numHiddenLayers=1, numNeurons=[5], activation=['ReLU'])
            network.fit(X_train, y_train)

            # Create a PSO object
            pso = PSO(num_particles=num_particles, max_iter=num_iterations, inertia=inertia, c1=c1, c2=c2, c3=c3, num_informants=num_informants)

            # Optimize the network using PSO
            start_time = time.time()
            network, best_fitness_list = pso.optimize(network, showFitnessUpdate=True)
            runtime = time.time() - start_time
            runtimeList.append(runtime)

            # add the best fitness of each iteration to the dataframe
            fitness_df.loc[j] = best_fitness_list

            # Evaluate the performance of the optimized network on test data
            predictions = network.predict(X_test)
            accuracy = get_accuracy(y_test, predictions)
            print("Best Fitness: ", best_fitness_list[-1])
            print("Accuracy on test data ", j, ":", accuracy)
            accuracyList.append(accuracy)
        
        # save the dataframe to a csv file named test_i.csv in a folder called Fitness_Data
        fitness_df.to_csv('Results/Results_Fitness/test_' + str(i) + '.csv')
        # save the accuracy of each run to the dataframe
        accuracy_df.loc[i] = accuracyList
        print("Average accuracy: ", np.mean(accuracyList))
        # save the runtime of each run to the dataframe
        runtime_df.loc[i] = runtimeList
        print("Average runtime: ", np.mean(runtimeList))

    # save the dataframe to a csv file named accuracy.csv in a folder called Results
    accuracy_df.to_csv('Results/accuracy.csv')
    print(accuracy_df)
    # save the dataframe to a csv file named runtime.csv in a folder called Results
    runtime_df.to_csv('Results/runtime.csv')
    print("\n")
    print(runtime_df)

    

def main():
    print("Which dataset do you want to use?")
    print("Options: [1] coursework, [2] iris")
    dataset = int(input())
    while(dataset not in [1, 2]):
        print("Invalid dataset. Please try again.")
        print("Options: [1] coursework, [2] iris")
        dataset = int(input())

    if dataset == 1:
        data_csv = pd.read_csv('data.csv')
        X = data_csv.iloc[:, :-1].to_numpy()
        y = data_csv.iloc[:, -1].to_numpy()
    elif dataset == 2:
        iris = load_iris()
        X = iris.data
        y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print("Creating a neural network...")

    print("How many hidden layers do you want in your network?")
    num_layers = int(input())

    neurons_per_layer = []
    activation_functions = []
    for i in range(num_layers):
        print("How many neurons do you want in hidden layer ", i+1, "?")
        num_neurons = int(input())
        print("What activation function do you want in layer ", i+1, "?")
        print("Options: [1] logistic, [2] ReLU, [3] leakyReLU, [4] tanh")
        activation = int(input())
        while(activation not in [1, 2, 3, 4]):
            print("Invalid activation function. Please try again.")
            print("Options: [1] logistic, [2] ReLU, [3] leakyReLU, [4] tanh")
            activation = int(input())
        
        if activation == 1:
            activation = 'logistic'
        elif activation == 2:
            activation = 'ReLU'
        elif activation == 3:
            activation = 'leakyReLU'
        elif activation == 4:
            activation = 'tanh'

        neurons_per_layer.append(num_neurons)
        activation_functions.append(activation)

    print("Creating a neural network with ", num_layers, " layers, ", neurons_per_layer, " neurons per layer, and ", activation_functions, " activation functions...")
    network = NeuralNetwork(num_layers, neurons_per_layer, activation_functions)
    network.fit(X_train, y_train)

    print("How many particles do you want in your PSO algorithm?")
    num_particles = int(input())
    print("How many iterations do you want in your PSO algorithm?")
    num_iterations = int(input())
    print("What inertia parameter do you want in your PSO algorithm?")
    inertia = float(input())
    print("What cognitive parameter do you want in your PSO algorithm?")
    cognitive = float(input())
    print("What social parameter do you want in your PSO algorithm?")
    social = float(input())
    print("What global parameter do you want in your PSO algorithm?")
    global_param = float(input())
    print("How many informants should each particle have?")
    num_informants = int(input())

    print('Do you want to take an average of multiple runs?')
    print('Options: [1] Yes, [2] No')
    average = int(input())

    print('How many runs do you want to average?')
    num_runs = int(input())

    runtimes = []
    accuracies = []

    for i in range(num_runs):
        print('\nRun ', i+1, ' of ', num_runs)

        network = NeuralNetwork(num_layers, neurons_per_layer, activation_functions)
        network.fit(X_train, y_train)

        # calculate runtime
        start_time = time.time()
        print("Training the network with PSO...")
        network.train_with_PSO(num_particles, num_iterations, inertia, cognitive, social, global_param, num_informants=num_informants)
        print("Training complete!")
        runtime = time.time() - start_time
        print("Runtime: ", runtime)
        runtimes.append(runtime)

        print("Evaluating the performance of the network on the test data...")
        y_pred = network.predict(X_test)
        accuracy = get_accuracy(y_test, y_pred)
        print("Accuracy on test data: ", accuracy)
        accuracies.append(accuracy)

        
    # Metrics
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report: ")
    print(classification_report(y_test, y_pred))


    

if __name__ == "__main__":
    # visualizeCourseworkData()
    # visualizeIrisData()
    # main_ANN_Test()
    # main_PSO_Test()
    # main_plot()
    # main_MLP_Test()
    coefficientTesting()
    # main()