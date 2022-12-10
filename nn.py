
from numpy import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('mnist_train.csv')

data = np.array(data)
np.random.shuffle(data)

m, n = data.shape

data_dev= data[0:1000].T
labels_dev = data_dev[0]
input_dev = data_dev[1:n].T
# input_dev= input_dev/255


data_train= data[0:1000].T
labels_train = data_train[0]
input_train = data_train[1:n]
input_train = input_train /255


    
def initialize():
    w1 = random.rand(10,784) -0.5
    b1 = random.rand(10,1) -0.5
    w2 = random.rand(10,10) -0.5
    b2 = random.rand(10, 1) -0.5
    return w1, b1, w2,b2


def relu(inputs):
    output = np.maximum(0,inputs)
    return output


def softmax(inputs):
    values = np.exp(inputs) / sum( np.exp(inputs) )

    return values

def forwardprop(w1,b1,w2,b2,input):
    z1 = w1.dot(input) + b1
    a1 = relu(z1)
    z2 = w2.dot(a1) + b2
    a2  = softmax(z2)
    return z1,a1,z2,a2

def one_hot(Y):
    oh_y = np.zeros((Y.size,Y.max()+1))
    oh_y[np.arange(Y.size),Y] = 1
    oh_y = oh_y.T
    return(oh_y)


def get_relu_derive(Z):
    return Z > 0


def backwarprop(z1,l1,z2,l2,w1,w2,inputs,output):
    onehot_y = one_hot(output)
    dz2 = l2 - onehot_y
    dw2 = 1/m* dz2.dot(l1.T)
    db2 = 1/m * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * get_relu_derive(z1)
    dw1 = 1/m * dz1.dot(inputs.T)
    db1 = 1/m * np.sum(dz1)
    return dw1,db1,dw2,db2

def update(w1,b1,w2,b2,dw1,db1,dw2,db2,a):
    w1 = w1 - a * dw1
    b1 = b1 - a * db1
    w2 = w2 - a * dw2
    b2 = b2 - a * db2
    return w1,b1,w2,b2


def predict(a2):
    return np.argmax(a2, 0)

def accuracy(predictions, labels):
    return np.sum(predictions == labels) / labels.size

def gradient_descent(input,output,iterations):
    
    aw1, ab1, aw2,ab2 = initialize()
    w1 = aw1
    b1 = ab1
    w2 = aw2
    b2 = ab2
    alpha = 5
    for i in range(iterations):
        z1,a1,z2,a2 = forwardprop(w1,b1,w2,b2,input)
        dw1,db1,dw2,db2 =  backwarprop(z1,a1,z2,a2,w1,w2,input,output)
        xw1,xb1,xw2,xb2 = update(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha)
        w1 = xw1
        b1 = xb1
        w2 = xw2
        b2 = xb2

        predictions = predict(a2)
        alpha = (5 - accuracy(predictions,output) )* (5 - accuracy(predictions,output) ) 
        # if i %10 == 0:
        #     print("iteration:", i)         
        #     print("alpha:", alpha)
        #     print("acccuracy:", accuracy(predictions,output))
    print("acccuracy:", accuracy(predictions,output))
    return w1,b1,w2,b2

w1,b1,w2,b2 = gradient_descent(input_train,labels_train,500)

        
def make_predictions(input,w1,b1,w2,b2):
    _, _, _,a2 = forwardprop(w1,b1,w2,b2,input)
    predictions = predict(a2)
    return(predictions)

def test(index,w1,b1,w2,b2):
    current_image = input_train[:,index,None]
    prediction = make_predictions(current_image,w1,b1,w2,b2)
    print("prediction:",prediction )         
    print("label:", labels_train[index])

    current_image = current_image.reshape((28,28))*255
    plt.gray()
    plt.imshow(current_image, interpolation="nearest")
    plt.show()

for i in range(10):
    test(i,w1,b1,w2,b2)