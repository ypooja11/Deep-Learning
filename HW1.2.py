import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)
print(train_data)
print(train_data.data.size())
print(test_data)
print(test_data.data.size())

from torch.utils.data import DataLoader
loaders = {
    'train' : torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=100, 
                                          shuffle=True, 
                                          num_workers=1),
}
loaders

import torch.nn as nn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU()                                          
        )        
        self.out = nn.Linear(32 * 7 * 7, 10)    
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)        
            x = x.view(x.size(0), -1)       
            output = self.out(x)
            return output, x    
cnn=CNN()
#print(cnn)
loss_func = nn.CrossEntropyLoss()   
loss_func

from torch import opti
optimizer = opti.Adam(cnn.parameters(), lr = 0.01)   
optimizer

from torch.autograd import Variable
num_epochs = 10
def train(num_epochs, cnn, loaders):
    
    cnn.train()
        
    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            b_x = Variable(images)  
            b_y = Variable(labels)   
            
            output = cnn(b_x)[0]               
            loss = loss_func(output, b_y)
            
            optimizer.zero_grad()           
            
            loss.backward()                             
            optimizer.step()                
            plt.plot(epoch,loss.item())
            plt.title('Training loss')
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))               
                pass
        
        pass
    
    
    pass
    plt.show()
        
train(num_epochs, cnn, loaders)

def test():
    cnn.eval()    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
            passprint('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)
            plt.plot(accuracy)
        plt.show()
    
    passtest()
