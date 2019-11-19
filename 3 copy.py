import pandas as pd 
from torch.nn import Linear, Module, MSELoss
dataset = pd.read_csv ('dataset.csv')
import torch

data = torch.tensor(dataset.values)
#print(data)
dataset.head()
Y = data[:, 5]
X = data[:, 0:5]
W = torch.rand(5,1)
#print(Y)
#print(X)
print("W.shape",W.shape)
print("X.shape=",X.shape)
print("Y.shape=",Y.shape)
#w = torch.rand([1, 5])
#print(w)
#print("XT",X.t().shape)
#print("X=",X.shape)
#print("Identity Matrix I =",torch.eye(5).shape)
#print("XT * X =",torch.mm(X.t(),X).shape)
#print("XT * X + I =",torch.add(torch.mm(X.t(),X),torch.eye(5)).shape)
#print("( XT * X + I ).INVERSE =",torch.inverse(torch.add(torch.mm(X.t(),X),torch.eye(5))).shape)
#print("( XT * X + I ).INVERSE * XT=",torch.mm(torch.inverse(torch.add(torch.mm(X.t(),X),torch.eye(5))),X.t()).shape)
W1 = torch.mm(torch.inverse(torch.add(torch.mm(X.t(),X),torch.eye(5))),X.t())
print(W1)
print(W1.shape)
print(Y.shape)
W2 = torch.mm(W1,Y)
print(W2)

#W=torch.mm(torch.inverse(torch.add(torch.mm(X.t(),X),torch.eye(5))),torch.mm(X.t(),Y))

#print("( XT * X + I ).INVERSE * XT * Y)=",torch.mm(torch.mm(torch.inverse(torch.add(torch.mm(X.t(),X),torch.eye(5))),X.t()),Y).shape)
print("Y=",Y)
print("W=",W)
#x_values = [i for i in range(11)]
#x_train = np.array(x_values, dtype=np.float32)
#x_train = x_train.reshape(-1, 1)

#y_values = [2*i + 1 for i in x_values]
#y_train = np.array(y_values, dtype=np.float32)
#y_train = y_train.reshape(-1, 1)


#from torch.autograd import Variable
#class linearRegression(torch.nn.Module):
#    def __init__(self, inputSize, outputSize):
#        super(linearRegression, self).__init__()
#        self.linear = torch.nn.Linear(inputSize, outputSize)
#
#    def forward(self, x):
#        out = self.linear(x)
#        return out

#def forward(x):
#    return torch.mm(x,w)
#
#def loss(x,y):
#    y_pred=forward(x)
#    return 