#%%
import torch
from time import time

a = torch.ones(1000)
b = torch.ones(1000)

a,b

# %%
start = time()
c = torch.zeros(1000)
for i in range(1000):
    c[i] = a[i] + b[i]
print(time() - start)

# %%
start = time()
d = a + b
print(time() - start)

# %%
# y = xw + b
a = torch.ones(3)
b = 10
print(a + b)

# %%
%matplotlib inline
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

# %%
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
ture_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
labels = true_w[0] * features[ : ,0] + true_w[1] * features[ : ,1]
labels += torch.tensor(np.random.normal(0,0.01, size=labels.size()), dtype=torch.float32)


# %%
print(features[0], labels[0])

# %%
def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

# %%
set_figsize()
plt.scatter(features[ : ,1].numpy(), labels.numpy(), 1)

# %%
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i+batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


# %%
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

# %%
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

# %%
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# %%
def linreg(X, w, b):
    return torch.mm(X, w) + b

# %%
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 /2

# %%
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size

# %%
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss


# %%
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        sgd([w,b], lr, batch_size)

        w.grad.data.zero_()
        b.grad.data.zero_()

    train_l = loss(net(features, w,b), labels)
    print('epoch %d, loss %f' % (epoch+1, train_l.mean().item()))

# %%
