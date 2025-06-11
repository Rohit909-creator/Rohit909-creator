#simple neural network
#Perceptron

import numpy as np



training_in = np.array([[0,0,1],
                        [1,1,1],
                        [1,0,1],
                        [0,1,1]])

labels = np.array([[0,1,1,0]]).T

np.random.seed(1)


synaptic_weights = np.random.random([3,1])


print("synaptic weights")
print(synaptic_weights)

def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    return x*(1-x)



for i in range(10000):

    input_layers = training_in

    outputs = sigmoid(np.dot(input_layers, synaptic_weights))
    error = labels - outputs

    adjustments = error*sigmoid_derivative(outputs)
    input_layer = np.transpose(input_layers)
    synaptic_weights += np.dot(input_layer,adjustments)


print('synapic_weights after training')
print(synaptic_weights)


print('Outputs after training')
print(outputs)
print("training complete")


while True:
    x1 = int(input(">>>"))
    x2 = int(input(">>>"))
    x3 = int(input(">>>"))
    

    inlayer = [x1,x2,x3]

    result = sigmoid(np.dot(inlayer,synaptic_weights))
    print(result)
    
