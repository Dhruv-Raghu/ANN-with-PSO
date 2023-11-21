import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import copy
from sklearn.datasets import load_iris

# np.random.seed(1911)

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
        outputs = []
        for neuron in self.layer:
            outputs.append(neuron.forward(inputs))
        return np.array(outputs).flatten()
    
    def updateWeights(self):
        iteration = 0
        for neuron in self.layer:
            neuron.weights = self.weights[iteration]
            neuron.bias = self.bias[iteration]
            iteration += 1
    
    def size(self):
        return self.num_neurons
    
class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.fitComplete = True

    def appendLayer(self, layer):
        self.layers.append(layer)

    def createLayer(self, num_neurons, activation):
        if self.fitComplete:
            if len(self.layers) == 0:
                num_inputs = self.X.shape[1]
            else:
                num_inputs = self.layers[-1].size()

            num_outputs = num_neurons

            # Initialize weights and bias in the Layer class
            layer = Layer(num_outputs, num_inputs, activation)
            self.layers.append(layer)
        else:
            print("Please provide input data and class labels first using <network>.fit(X, y) first")

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
    
    def get_params(self):
        params = []
        for layer in self.layers:
            params.append(layer.weights.flatten())
            params.append(layer.bias.flatten())
        return np.concatenate(params)


class PSO:
    def __init__(self, num_particles, max_iter, inertia_param, cognitive_param, social_param, global_param):
        self.num_particles = num_particles
        self.particles = []
        self.max_iter = max_iter
        self.alpha = inertia_param      # determines how much of the particle's current velocity should be retained
        self.beta = cognitive_param     # determines how much of the particle's best position should be retained
        self.gamma = social_param       # determines how much of the informants's best position should be retained
        self.delta = global_param       # determines how much of the swarm's best position should be retained

    def optimize(self, network):
        self.createParticles(network)

        best_particleFitness = 0
        best_network = None
        global_best = None

        for x in range(self.max_iter):
            print("Iteration: ", x)
            print("Best fitness: ", best_particleFitness)

            for particle in self.particles:

                # dynamically change inertia, cognitive, social, and global parameters with iteration
                if x > 0.25 * self.max_iter:
                    self.alpha = max(0.8, self.alpha - 0.1)  # Reduce exploration
                    self.beta = max(0.8, self.beta - 0.1)
                    self.gamma = max(0.6, self.gamma - 0.1)
                    self.delta = min(0.6, self.delta + 0.1)  # Increase exploitation
                elif x > 0.5 * self.max_iter:
                    self.alpha = max(0.5, self.alpha - 0.1)
                    self.beta = max(0.6, self.beta - 0.1)
                    self.gamma = max(0.4, self.gamma - 0.1)
                    self.delta = min(1.2, self.delta + 0.1)
                elif x > 0.75 * self.max_iter:
                    self.alpha = max(0.4, self.alpha - 0.1)
                    self.beta = max(0.4, self.beta - 0.1)
                    self.gamma = max(0.2, self.gamma - 0.1)
                    self.delta = min(1.5, self.delta + 0.1)


                
                if x > 0:
                    self.updateInformantBest(particle)
                    particle.updateVelocity(self.alpha, self.beta, self.gamma, self.delta, global_best)
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

    def findNeighbours(self, particle, k=3):
        distances = [np.linalg.norm(particle.position.get_params() - other.position.get_params()) for other in self.particles if other != particle]
        indices = np.argsort(distances)[:k]
        neighbours = [self.particles[i] for i in indices]
        return neighbours
    
    def updateInformantBest(self, particle):
        neighbours = self.findNeighbours(particle)
        best_neighbour = max(neighbours, key=lambda x: x.fitness)
        if best_neighbour.fitness > particle.informant_best_fitness:
            particle.informant_best = copy.deepcopy(best_neighbour)
            particle.informant_best_fitness = copy.deepcopy(best_neighbour.fitness)
    
class Particle:
    def __init__(self, network, bound=5.0):
        self.position = copy.deepcopy(network)
        self.best_position = network
        self.informant_best = None
        self.informant_best_fitness = 0
        self.fitness = 0
        self.best_fitness = 0
        self.velocities = []
        self.initializeVelocity()
        self.bound = bound

        # self.num_informants = np.random.randint(3, 7) # will implement informants later

    def initializeVelocity(self):
        for layer in self.position.layers:
            weights_velocity = np.random.randn(*layer.weights.shape) * 0.01
            bias_velocity = np.random.randn(*layer.bias.shape) * 0.01
            self.velocities.append((weights_velocity, bias_velocity))

    def updateVelocity(self, alpha, beta, gamma, delta, g_best):
        # alpha -> inertia_param      # determines how much of the particle's current velocity should be retained
        # beta -> cognitive_param     # determines how much of the particle's best position should be retained
        # gamma -> social_param       # determines how much of the informants's best position should be retained
        # delta -> global_param       # determines how much of the swarm's best position should be retained
        # g_best -> best performing network in the swarm

        # Update velocities element-wise
        for i, (weights_vel, bias_vel) in enumerate(self.velocities):
            weights_update = alpha * weights_vel + beta * np.random.uniform(0, 1) * (self.best_position.layers[i].weights - self.position.layers[i].weights) + gamma * np.random.uniform(0,1) * (self.informant_best.position.layers[i].weights - self.position.layers[i].weights) + delta * np.random.uniform(0, 1) * (g_best.layers[i].weights - self.position.layers[i].weights)
            bias_update = alpha * bias_vel + beta * np.random.uniform(0, 1) * (self.best_position.layers[i].bias - self.position.layers[i].bias) + gamma * np.random.uniform(0,1) * (self.best_position.layers[i].bias - self.informant_best.position.layers[i].bias) + delta * np.random.uniform(0, 1) * (g_best.layers[i].bias - self.position.layers[i].bias)

            # Update velocities
            self.velocities[i] = (weights_update, bias_update)

    def updatePosition(self):
        for i, layer in enumerate(self.position.layers):
            # Update weights and apply reflective bounding
            layer.weights = layer.weights + self.velocities[i][0]
            layer.weights = self.reflectiveBound(layer.weights, -self.bound, self.bound)  # Reflective bounding

            # Update biases and apply reflective bounding
            layer.bias = layer.bias + self.velocities[i][1]
            layer.bias = self.reflectiveBound(layer.bias, -self.bound, self.bound)  # Reflective bounding

            layer.updateWeights()
            

    def evaluateFitness(self):
        y_pred = self.position.predict(self.position.X)
        self.fitness = get_accuracy(self.position.y, y_pred)
        if self.fitness > self.best_fitness:
            local_best = self.position
            local_best_fitness = self.fitness
            self.best_position = local_best
            self.best_fitness = local_best_fitness

    def reflectiveBound(self, value, lower_bound, upper_bound):
        # Reflective bounding for an array of values
        for i in range(len(value)):
            value[i] = np.where(value[i] < lower_bound, 2 * lower_bound - value[i], value[i])
            value[i] = np.where(value[i] > upper_bound, 2 * upper_bound - value[i], value[i])
        return value


def get_accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

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

    # take an average of 10 runs
    accuracyList = []
    for i in range(10):
        # Create a neural network
        network = NeuralNetwork()
        network.fit(X_train, y_train)
        network.createLayer(8, Activation('ReLU'))
        # network.createLayer(20, Activation('leakyReLU'))
        network.createLayer(8, Activation('ReLU'))
        network.createLayer(3, Activation('logistic'))

        # Initialize PSO with test parameters
        num_particles = 30
        max_iter = 100
        inertia_param = 1.0
        cognitive_param = 1.2
        social_param = 1.2
        global_param = 1.5

        pso = PSO(
            num_particles=num_particles,
            max_iter=max_iter,
            inertia_param=inertia_param,
            cognitive_param=cognitive_param,
            social_param=social_param,
            global_param=global_param
        )

        # Optimize the neural network using PSO
        optimized_network = pso.optimize(network)

        # Evaluate the performance of the optimized network on test data
        predictions = optimized_network.predict(X_test)
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

    


def paramSearch():
    pass

if __name__ == "__main__":
    # visualizeCourseworkData()
    # visualizeIrisData()
    # main_ANN_Test()
    main_PSO_Test()
    # main_MLP_Test()