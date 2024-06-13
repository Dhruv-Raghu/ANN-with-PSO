# F20BC_Coursework
ANN implementation from scratch with PSO optimization for training

## Implementation
The ANN implementation consists of four classes: `Activation`, `Neuron`, `Layer`, and `Network`. The weights of each layer are initialized using He initialization, known for enhancing convergence in neural networks. The input and output layers are automatically generated based on the input data and class labels. By default, the output layer employs the logistic activation function, as the network is intended for classification tasks.

The `forward` method in the neural network object processes a row of input data through the network, storing the raw output from the final layer. The `predict` method passes input data row by row through the `forward` function, converting the raw neuron outputs into class predictions. The `train_with_PSO` function initializes a PSO object with specified parameters, aiming to tune the network's weights and biases to accurately classify the input data.

The PSO implementation, which includes informants, is divided into two classes: the `Particle` class and the `PSO` class. In this implementation, each particle's position represents a neural network object. When particles are created, the weights and biases of the neural network object are randomized to ensure proper exploration of the search space, preventing all particles from starting from the same point. The particle fitness is evaluated using cross-entropy loss via the loss function in the neural network class. The algorithm aims to return the network with the lowest fitness. With minor modifications, particle fitness can be evaluated using accuracy calculated on the training set instead of cross-entropy loss.

The `Particle` class also handles updating the velocity and position throughout the algorithm. Reflective bounds are applied to the particles to avoid large positional changes.

The primary function of the PSO algorithm is the `optimize` function. This function iteratively updates the particles' positions and velocities, and updates the global and informant bests if necessary. The PSO algorithm includes informants, determined by identifying the 'k' closest neighbors to a given particle based on fitness comparison. The `optimize` function supports dynamic parameter adjustments, such as linearly decreasing inertia weight. Additionally, code is available to modify multiple parameters based on iteration, which can be useful for complex problems to balance exploration and exploitation for optimal results.


## Experimental Investigation
As explained in the Background Theory section, the Artificial Neural Network (ANN) and Particle Swarm Optimization (PSO) algorithm have various parameters that influence how data relationships are formed and how the neural network learns. Instead of randomly adjusting the three ANN parameters and seven PSO parameters, which would be inefficient and time-consuming, a structured approach is devised to optimize a few parameters at a time until the whole network is fully optimized.

### PSO Parameters
Before analyzing the impact of ANN parameters on the network output, it is crucial to optimize the PSO algorithm.

#### Cognitive, Social, and Global Weights
The first test focuses on the three coefficients: cognitive, social, and global parameters. A common practice is to configure these coefficients so that their sum equals 4. Based on this guideline, six parameter sets were created, each focusing on a distinct combination of these parameters.

<img width="500" alt="image" src="https://github.com/Dhruv-Raghu/F20BC_Coursework/assets/86726252/58a1277c-84d0-4336-a747-f1b6d430f0a0">

Given the limited size of the banknote authentication dataset (4 features, binary classification), a neural network with a single hidden layer, 5 neurons, and a ReLU activation function is created as a constant. The swarm size is set to 25 particles, the number of informants is set to 4, the number of iterations is kept at 25 due to the dataset's simplicity, and the inertia is set to 0.5 to balance exploration and exploitation.

The network is trained using the parameters from the first test set. The accuracy of the test set is computed and stored in a list. Given the stochastic nature of the PSO algorithm, each set is tested 15 times, and the average accuracy is calculated. This process is repeated for each test set, allowing a comparison of the accuracies from each run to identify the optimal parameter set.

#### Inertia
To determine the optimal value for the inertia parameter, the constants from the previous test remain unchanged. However, this time, the algorithm's coefficients are set to the optimal values identified previously, and only the inertia parameter is varied. Five inertia values (0.2, 0.5, 0.8, and 1.0) are tested, with the network undergoing 15 runs for each inertia weight. The average accuracy over 15 runs is calculated and compared to choose the optimal weight.

#### Swarm Size and Number of Iterations
Increasing the swarm size and the number of iterations generally enhances the performance of the neural network. However, excessive increases might lead to diminishing returns due to added complexity and increased runtime. Five test sets are created to determine the most effective balance for the binary classification dataset, exploring the impact of swarm size and iterations on accuracy and runtime.

<img width="500" alt="image" src="https://github.com/Dhruv-Raghu/F20BC_Coursework/assets/86726252/1f8c73e7-26ee-4c36-9c54-c680cc7f5b4d">

Note: The choice of swarm sizes and iterations is determined based on the simplicity of the banknote authentication dataset. What might be a large swarm size for this dataset could be perceived as a small swarm size for a more complicated dataset with a larger search space.

#### Informants
The ideal number of informants typically falls within the range of 4 to 8. To test this, the network is trained using 2, 6, and 10 informants. If 6 proves to be the optimal number, additional testing can be conducted to refine the optimal number within the 4 to 8 range.

### ANN Parameters
When building the ANN, it's often found that having three layers works well. The network was tested with 1, 3, and 5 hidden layers, and it was determined that a single hidden layer was sufficient to achieve high accuracy on the test set. Adding more layers increased complexity and runtime without substantially improving accuracy. For the neurons in the hidden layers, a good rule of thumb is to use 70% to 90% of the nodes from the input layer, which helps the network understand patterns without becoming overly complicated. Various activation functions like ReLU, Sigmoid, Tanh, and leaky ReLU were tested to determine the most effective one.

## Result Discussion

### PSO Parameters
#### Cognitive, Social, and Global Weights
Figure 1 illustrates the accuracy fluctuations of the PSO algorithm across multiple runs. Each test set uses a distinct combination of cognitive, social, and global parameters while keeping other parameters constant.

<img width="500" alt="image" src="https://github.com/Dhruv-Raghu/F20BC_Coursework/assets/86726252/f66df033-aee9-4052-9a2c-aa0c72b2c6cf">
<img width="500" alt="image" src="https://github.com/Dhruv-Raghu/F20BC_Coursework/assets/86726252/540a64d7-ef1c-4a05-aff0-d9babe88fcc8">

Test Set 1, focusing on the global best, shows high variance in accuracy due to its exploitation around the global maximum, risking convergence to local minima. Test Sets 2 and 3, exploring personal and swarm bests respectively, also exhibit this issue. Test Set 4 balances personal and swarm bests over the global best, resulting in more stable accuracy. Test Set 5, giving equal weight to all parameters, achieves the highest minimum accuracy and a perfect run. Test Set 6 increases the weight of the global parameter, achieving the highest average accuracy with minimal variance, making it the optimal set for this dataset.

#### Inertia
Testing four inertia values (0.2, 0.5, 0.8, and 1.0), 0.5 emerged as the best, offering the highest average, maximum, and minimum accuracies. High inertia values hinder quick convergence, while low values risk early convergence to local minima. The balance provided by 0.5 allows effective exploration and convergence.

<img width="500" alt="image" src="https://github.com/Dhruv-Raghu/F20BC_Coursework/assets/86726252/0f067f46-cd5c-4896-82c3-a002a3bbd749">


#### Swarm Size and Number of Iterations
Balancing swarm size and iterations is crucial for performance and complexity. Test Set 3, with 25 particles and 25 iterations, offers the best balance, reducing runtime significantly while maintaining high accuracy.

<img width="500" alt="image" src="https://github.com/Dhruv-Raghu/F20BC_Coursework/assets/86726252/10bc63d9-65f6-42c7-aca3-48609ca6955e">

<img width="500" alt="image" src="https://github.com/Dhruv-Raghu/F20BC_Coursework/assets/86726252/f5b4ee34-fabc-4c50-bae5-ab3cdc6d3c55">

Test Set 3 achieves a near-optimal accuracy with significantly reduced runtime compared to Test Set 2.

#### Number of Informants
Testing 2, 6, and 10 informants, it was found that fewer informants yield better average accuracy due to the simplicity of the banknote authentication dataset.

<img width="500" alt="image" src="https://github.com/Dhruv-Raghu/F20BC_Coursework/assets/86726252/20936cc8-b3d4-4ff9-8033-101fb27f6c3e">

## Conclusion
The Particle Swarm Optimization (PSO) algorithm effectively enhanced the performance of the Artificial Neural Network (ANN), consistently achieving accuracy above 90% on the test set with optimized parameters. However, PSO's convergence was slower compared to the backpropagation algorithm. The simplicity of the banknote authentication dataset (CW dataset) constrained the PSO's exploratory capabilities, favoring global best exploitation and minimizing the impact of linearly decreasing inertia weights. It was also observed that fewer informants, particles, and iterations were sufficient for this straightforward dataset.

Future work should focus on testing the PSO-ANN combination on more complex datasets to fully explore the algorithm's potential. Additionally, implementing adaptive parameters that adjust dynamically based on the algorithm's state, such as particle positions and velocities, could further improve performance and efficiency.

## References
[1] Effects of hidden layers on the efficiency of neural networks | IEEE ..., https://ieeexplore.ieee.org/document/9318195 (accessed Nov. 24, 2023). 
[2] International Journal of Engineering Trends and Technology ..., http://www.ijettjournal.org/volume-3/issue-6/IJETT-V3I6P206.pdf (accessed Nov. 24, 2023). 
[3] J. Garcia-Nieto and E. Alba, “Why Six informants is optimal in PSO,” Proceedings of the 14th annual conference on Genetic and evolutionary computation, 2012. doi:10.1145/2330163.2330168 







