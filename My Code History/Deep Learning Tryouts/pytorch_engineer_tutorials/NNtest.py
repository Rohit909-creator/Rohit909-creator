import numpy as np


training_set = np.array([[14,0],
                         [1,1],
                         [11,0],
                         [6,1]])
labels = np.array([[0,1,0,1]]).T

synaptic_weights = np.random.random([2,1])

print(training_set)

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return x*(1-x)


for i in range(20000):
    
    input_layers = training_set
    output = sigmoid(np.dot(input_layers,synaptic_weights))
    error = labels - output
    adjustments = error*sigmoid_derivative(output)
    input_layers = np.transpose(input_layers)
    #new synaptic weights
    synaptic_weights += np.dot(input_layers,adjustments)


print(output)

