import pandas as pd 
from torch.nn import Linear, Module, MSELoss
dataset = pd.read_csv ('dataset.csv')
import torch

data = torch.tensor(dataset.values)

dataset.head()
Y = data[:, 5]
X = data[:, 0:5]
W = torch.rand(5,1)

# for Lambda = 1

lam=1
W = torch.mm(torch.mm(torch.inverse(torch.add(torch.mm(X.t(),X),torch.eye(5))),X.t()),Y.view([1502,1]))
#print("W=",W)
fx=torch.mm(X,W)
print(fx)
print(fx.shape)
criterion = torch.nn.MSELoss()

loss = 0.5 * criterion(fx,Y.view([1502,1]))
loss = loss + 0.5 * lam * torch.mm(W.t(),W)
print(loss)


# for Lambda = 0.1

lam=0.1
W = torch.mm(torch.mm(torch.inverse(torch.add(torch.mm(X.t(),X),torch.eye(5))),X.t()),Y.view([1502,1]))
#print("W=",W)
fx=torch.mm(X,W)
print(fx)
print(fx.shape)
criterion = torch.nn.MSELoss()

loss = 0.5 * criterion(fx,Y.view([1502,1]))
loss = loss + 0.5 * lam * torch.mm(W.t(),W)
print(loss)

# for Lambda = 100

lam=100
W = torch.mm(torch.mm(torch.inverse(torch.add(torch.mm(X.t(),X),torch.eye(5))),X.t()),Y.view([1502,1]))
#print("W=",W)
fx=torch.mm(X,W)
print(fx)
print(fx.shape)
criterion = torch.nn.MSELoss()

loss = 0.5 * criterion(fx,Y.view([1502,1]))
loss = loss + 0.5 * lam * torch.mm(W.t(),W)
print(loss)