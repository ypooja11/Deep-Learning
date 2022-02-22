import torch
from torch import nn
import numpy as np
class Network(nn.Module):
    def __init__(self):
        self.input = x
        self.y = y
        self.weights_in = np.random.uniform(-1, 1,(self.y.shape[1],self.input.shape[1]))
        self.weights_out = np.random.uniform(-1, 1,(self.y.shape[1],self.input.shape[1]))     
        self.output     = np.random.uniform(-1, 1,self.y.shape)
        
def forward(self,fnc):
    self.output[fnc] = np.sum(self.weights_out*self.input[fnc]*np.sin(np.pi*self.input[fnc]*self.weights_in),axis = 1)
   
def backprop(self,fnc):
        error = np.square(self.y[fnc]-self.output[fnc])
        error_output = self.y[fnc]-self.output[fnc]
        output_weights_in = self.weights_out * np.square(self.input[fnc]) * np.pi * np.cos(np.pi*self.input[fnc]*self.weights_in)
        output_weights_out = self.input[fnc]*np.sin(np.pi*self.input[fnc]*self.weights_in)
        weights_in = np.dot(error_output,output_weights_in)
        weights_out = np.dot(error_output,output_weights_out)
        self.weights_in += weights_in*0.05
        self.weights_out += weights_out*0.05
        
 def predict(self,ip):
        predictions = []
        for elm in ip:
            predictions.append(np.sum(self.weights_out*elm*np.sin(np.pi*elm*self.weights_in),axis = 1).tolist())
        return np.array(predictions)
    
def save_weights(self,dir_in = './weights_in.npy',dir_out = './weights_out.npy'):
        np.save(dir_in,self.weights_in)
        np.save(dir_out,self.weights_out)
        
  def import_weights(self,dir_in = './weights_in.npy',dir_out = './weights_out.npy'):
        self.weights_in = np.load(dir_in)
        self.weights_out = np.load(dir_out)
x = np.array([[0.01,0.1,0.01],[0.01,0.99,0.99],[0.99,0.99,0.01],[0.99,0.99,0.01],[0.9,0.9,0.9]])
y = np.array([[0.01,0.01,0.01],[0.01,0.99,0.99],[0.01,0.99,0.99],[0.01,0.99,0.99],[0.99,0.99,0.01]])
nn = NeuralNetwork(x,y)
#TRAIN NETWORK FOR 2000 TIMES.
for gen_cnt in range(2000):
    for cnt in range(5):
        nn.feedforward(cnt)
        nn.backprop(cnt)
#PREDICT THE TEST DATA
predictions = nn.predict(np.array([[0.01,0.2,0.2],[0.9,0.1,0.95]]))
print('Predictions:\n',np.around(predictions),'\nExpected:\n',[[0,0,0],[0,1,1]])
