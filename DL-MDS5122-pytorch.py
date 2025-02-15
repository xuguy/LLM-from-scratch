import numpy as np
import torch

# randomly initialize a tensor:
random_tensor = torch.rand(3,3)
'''
random_tensor:
tensor([[0.0614, 0.1658, 0.6409],
        [0.5253, 0.7213, 0.3206],
        [0.5358, 0.5630, 0.7740]])
'''
random_tensor.dtype # torch.float32
random_tensor.device # device(type='cpu')


# tensor and numpy conversion
t = torch.ones(5)
print(f't: {t}')
n = t.numpy()
print(f'n: {n}')
t = torch.from_numpy(n)
print(f't: {t}')

# change in tensor will simlutaneously point to tensor:
t.add_(1)
print(f't: {t}')
print(f'n: {n}')

# change in numpy will also point to tnesor
np.add(n, 1, out = n)
print(f't: {t}')
print(f'n: {n}')


# mathematical operation
torch.manual_seed(1337)
r = (torch.rand(2, 2) - 0.5) * 2 # values between -1 and 1
print('A random matrix, r:')
print(r)

# Common mathematical operations are supported:
print('\nAbsolute value of r:')
print(torch.abs(r))

# ...as are trigonometric functions:
print('\nInverse sine of r:')
print(torch.asin(r))

# ...and linear algebra operations like determinant and singular value decomposition
print('\nDeterminant of r:')
print(torch.det(r))
print('\nSingular value decomposition of r:')
print(torch.svd(r))
torch.svd(r).U # to extract matrix: .S .V

# ...and statistical and aggregate operations:
print('\nAverage and standard deviation of r:')
print(torch.std_mean(r))
print('\nMaximum value of r:')
print(torch.max(r))

# ============================================
# Create a tensor and set requires_grad to True, default False, because it save ram
x = torch.tensor([3.0], requires_grad=True)

print(x.grad)
y = x**2
y.backward()
print(x.grad) # tensor([6.])

# more complicated cases
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

# more complex function:
Q = 3*a**3 - b**2
print(Q) # which is a (2,) tensor/vector
Q.backward() # error:
Q.backward(torch.ones_like(Q))
a.grad
b.grad
'''
this is because Q is a vecotor we cannot implicitly create grad on a vector: partial derivitive 只能对标量进行，不可以对变量进行。 we should pass a vecotr to backward()
'''
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)
a.grad
b.grad
print(9*a**2 == a.grad) # grad a cal element-wise

'''
下面来看一个例子，个例子叠了3层：(x1,x2,x3)->(y1,y2,y3)->f(y1,y2,y3). 注意，我们求偏导，永远只能做标量对某个变量的导数
'''

x1 = torch.tensor(1, requires_grad=True, dtype=torch.float)
x2 = torch.tensor(2, requires_grad=True, dtype=torch.float)
x3 = torch.tensor(3, requires_grad=True, dtype=torch.float)

x = torch.tensor([x1, x2, x3])

# initialize y
y=torch.randn(3)
# define y w.r.t x
y[0] = x1*x2*x3
y[1] = x1 + x2 + x3
y[2] = x1 + x2*x3

# 此时，假如说有任意一个函数A()，A将y作为自变量输入，我们不需要知道A的具体形式，就可以求出A关于x的grad，我们只需要给出gradient参数的具体值就可以了。典型的例子就是，神经网络前面的层一通运算，最后计算error，这个error就是A，计算出error的具体值以后，我们就可以求出error对所有权重的grad

# 下面我们传入[0.1, 0.2, 0.3]
y.backward(torch.tensor([0.1, 0.2, 0.3], dtype = torch.float))
print(x1.grad, x2.grad, x3.grad)

# 当gradient的参数都等于1时，等价于
y.sum().backward()
# 这也很符合我们的直觉，A = x1 + x2 + x3 的grad就是 [1, 1, 1]


'''
向量backward()特殊之处：
1. 一个多元函数f(x1, x2, ...)，接受一个向量作为输入，输出一个标量，那么parf/parx 就是一个和输入向量具有相同形状的向量
2. 一个向量函数（接受一个向量作为输入，同时输出一个向量），这个函数的grad就是一个mxn的矩阵

标量对向量求导，本质上时函数对各个自变量求导：标量关于向量的梯度时向量，向量对向量的梯度是矩阵，这个梯度矩阵矩阵可以用雅可比矩阵表示。

当backward中的gradient参数的元素都为1时，此时相当于求y.sum()的backward
'''

# simple nn
import torch.nn as nn
import torch.optim as optim

# define a simple NN
class SimpleNN(nn.Module):
    def __init__(self):
        '''
        check:
        https://blog.csdn.net/luobinrobin/article/details/144473933
        equivalent to super().__init__(), 这是因为python3的改进使得我们不必写那么多，但是为了兼容以前的版本，这里还是写成了super(SimpleNN, self).__init__()
        '''
        super(SimpleNN, self).__init__()
        # super().__init__()
        self.fc1 = nn.Linear(2, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
'''
super(Net, self).__init__()是指首先找到SimpleNN的父类（比如是类nn.Module），然后把类SimpleNN的对象self转换为类nn.Module的对象，然后“被转换”的类nn.Module对象调用自己的init函数，其实简单理解就是子类把父类的__init__()放到自己的__init__()当中，这样子类就有了父类的__init__()的那些东西。

当然，如果子类初始化的逻辑与父类的不同，不使用父类的方法，自己重新初始化也是可以的。

这样做的好处是确保nn.Module的初始化逻辑得以执行，从而使SimpleNN类能够继承nn.Module的所有属性和方法，成为一个有效的PyTorch神经网络模块‌.

通过这种方式，SimpleNN类可以在不重复编写nn.Module的初始化代码的情况下，利用nn.Module提供的功能，如参数管理、前向传播等。这种机制使得代码更加简洁和可维护‌.
'''

# Create models, loss functions and optimizers
torch.manual_seed(1337)
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# input and target
inputs = torch.tensor([[1.0, 2.0]], requires_grad=True) #2D tensor
targets = torch.tensor([[0.0]], requires_grad=False)

# Train loop
for epoch in range(100):
    # forward propagation
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backpropagation and optimization
    optimizer.zero_grad()
    #Before performing backpropagation, the gradients need to be cleared.
    #PyTorch will increase the gradient value at each gradient accumulation,

    loss.backward()
    #The gradient of the model parameters with respect to the loss is automatically calculated.

    optimizer.step()
    #Perform a one-step gradient descent algorithm, update the weights of the model based on the calculated gradient to reduce the loss.

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item()}')

# print result
for name, param in model.named_parameters():
    print(f'Parameter {name}: {param.data}')

# now proceed to a more sophistcated one:
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

dataset = pd.read_csv('dl-dataset/DL-tutor1-diabetes.csv') #https://github.com/plotly/datasets/blob/master/diabetes.csv

#取出前8个features（该数据集一共就只有8个features）
X = dataset.iloc[:, :8]
#取出标签,0/1
y = dataset.iloc[:, -1]

# 数据转换为tensor
X = torch.tensor(X.to_numpy(), dtype=torch.float32)
y = torch.tensor(y.to_numpy(), dtype=torch.float32).reshape(-1, 1)

# define model: a fully connected feedforward nn
'''
1. The model expects rows of data with 8 variables （数据集有8个features） (the first argument at the first layer set to 8)
2. The first hidden layer has 12 neurons, followed by a ReLU activation function
3. The second hidden layer has 8 neurons, followed by another ReLU activation function
4. The output layer has one neuron, followed by a sigmoid activation function

'''
class PimaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(8, 12)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(12, 8)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(8, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x

model = PimaClassifier()
print(model)

'''
PimaClassifier(
  (hidden1): Linear(in_features=8, out_features=12, bias=True)
  (act1): ReLU()
  (hidden2): Linear(in_features=12, out_features=8, bias=True)
  (act2): ReLU()
  (output): Linear(in_features=8, out_features=1, bias=True)
  (act_output): Sigmoid()
)
'''

#%%

# a less-verbose way:
model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid())


print(model)
'''
Sequential(
  (0): Linear(in_features=8, out_features=12, bias=True)
  (1): ReLU()
  (2): Linear(in_features=12, out_features=8, bias=True)
  (3): ReLU()
  (4): Linear(in_features=8, out_features=1, bias=True)
  (5): Sigmoid()
)
'''
# prepare for training
loss_fn = nn.BCELoss()  # binary cross entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 10

# training loop
for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        
        # 注意这列的loss_fn之前定义过
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()#Before performing backpropagation, the gradients need to be cleared. If you don't clear the gradients using,
        #they will keep accumulating, which can lead to incorrect gradient updates and ultimately incorrect model training.
        loss.backward() #The gradient of the model parameters with respect to the loss is automatically calculated.
        optimizer.step() #update the weights of the model based on the calculated gradient to reduce the loss.
    print(f'Finished epoch {epoch}, latest loss {loss}')

'''
经过多次实验我们发现，最后loss在loop = 100， lr=0.001的情况下会稳定到0.3左右
'''

# %%

# 用training set评估模型
# with语句切换eval模式
with torch.no_grad():
    y_pred = model(X) # model(X): forward并输出预测结果(sigmoid)

'''
# y_pred.round(): 超过0.5的1，小于等于0.5的0
tmp = torch.tensor(0.5, dtype=torch.float32)
tmp.round() # 0
tmp = torch.tensor(0.501, dtype=torch.float32)
tmp.round() # 1
'''
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")# loss~0.376, Accuracy 0.76953125


# %%

# about tensor
tmp = [1,2,3,4,5,5]
t = torch.tensor(tmp)
t.dtype # torch.int64

# add dimension or unsqueeze
tmp = torch.tensor([[1,2,3],[4,5,6]]) # shape (2,3)
tmp[None].shape # shape (1,2,3)
tmp.unsqueeze(dim = 0).shape # shape (1,2,3), equivalent to [None] with param dim = 0

# lets pretend we have an image of 5x5
img_t = torch.randn(3,5,5) # channels, rows, columns
weights = torch.tensor([0.2126, 0.7152, 0.0722]) # to cal brightness, weights of R,G,B
batch_t = torch.randn(2,3,5,5) # batch, channels, rows, columns

im_gray_naive = img_t.mean(dim=-3)
batch_gray_naive = batch_t.mean(dim=-3)
im_gray_naive.shape, batch_gray_naive.shape

unsqueezed_weights = weights.unsqueeze(dim=-1).unsqueeze_(dim=-1) #shape: (3,1,1), the last .unsqueeze_ with underscore is equivalent to .unsqueeze() without one
img_weights = (img_t*unsqueezed_weights)
batch_weights = (batch_t*unsqueezed_weights)
img_gray_weighted = img_weights.sum(dim=-3) # the -3 dim: channel
batch_gray_weighted = batch_weights.sum(dim = -3)
batch_weights.shape, batch_t.shape, unsqueezed_weights.shape

# add name to dimension:
weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names = ['channels'])
weights_named # tensor([0.2126, 0.7152, 0.0722], names=('channels',))

# add names to tensor dim where already have names
img_named = img_t.refine_names(... , 'channels', 'rows', 'columns')
batch_named = batch_t.refine_names(... , 'channels', 'rows', 'columns')

# out: img named torch.Size([3, 5, 5]) ('channels', 'rows', 'columns')
print('img named', img_named.shape, img_named.names)
print('batch named', batch_named.shape, batch_named.names)

# align dim with names:
# weights 只有一个dim，name是channels，而img_named有3个dim，name分别是'channels' 'rows' 'columns'
weights_aligned = weights_named.align_as(img_named)
# align_as 把broadcast需要的步骤：unsqueeze自动化了
weights_aligned.shape, weights_aligned.names
# out: (torch.Size([3, 1, 1]), ('channels', 'rows', 'columns'))

gray_named = (img_named*weights_aligned).sum(dim = 'channels')
gray_named.shape, gray_named.names
# out: (torch.Size([5, 5]), ('rows', 'columns'))

# drop names to go back to operation of unnamed dim:
gray_plain = gray_named.rename(None)
gray_plain.shape, gray_plain.names
# out: (torch.Size([5, 5]), (None, None))

# tensor vs storage:
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storage()
# change the storage will change the tensor

# IN-PLACE operation:
a = torch.ones(3,2)
a.zero_()
a # become zero with operation trailing underscore

# offset, stride, size
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
second_point = points[1] # [5.0, 3.0]
second_point.storage_offset() # out: 2
second_point.size() # out: torch.Size([2])
second_point.shape # same as above
'''
accessing an element i, j in a 2D tensor =
storage_offset + stride[0] * i + stride[1] * j 
'''

points.stride() #(2, 1)
second_point.stride() # (1,): 第二个点是一个1维向量，因此只有一个维度而不是2两，他的下一个点的stride距离第一个点为
second_point.storage_offset() # still be 2, 因为subvector和原vector具有相同的storage
# 因此，改变subvector，会直接改变原vector，最好不要这样做
# 我们应该用.clone() 取修改subvector，避免对原vector产生变化

second_point = points[1].clone()
second_point.storage_offset() # become 0: new storage

# transposing without copying
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points

points_t = points.t()
points_t
points.storage().data_ptr() == points_t.storage().data_ptr()

#  the storage holds the elements in the tensor sequentially row by row
points.stride() # (2, 1)
points_t.stride() # (1, 2)

# transposing multi dim
some_t = torch.ones(3,4,5)
transpose_t = some_t.transpose(0, 2) # transpose dim 0 and 2
some_t.shape # torch.Size([3, 4, 5])
transpose_t.shape # torch.Size([5, 4, 3])

some_t.stride() # (20, 5, 1)
transpose_t.stride() # (1, 5, 20)

# contiguous
'''
A 'tensor'(hence not storage) whose 'values' are laid out in the storage starting from the rightmost dimension onward (that is, moving along rows for a 2D tensor) is defined as contiguous.
'''
points.is_contiguous() # true

points_t.is_contiguous() # false, whu? check def of contiguous above
points_t_cont = points_t.contiguous()
points_t_cont.is_contiguous() # True

points_t_cont.stride()
points_t_cont.storage()
points.storage() # not same as points_t_cont, because storage been reshuffled in order for elements to be laid out row-by-row, and hence the stride will also be changed to reflect the new layout


# save tensor
with open('dl-dataset/ourpoints_tmp.t', 'wb') as f:
    torch.save(points, f)

# reload tensor
with open('dl-dataset/ourpoints_tmp.t', 'rb') as f:
    points_reload = torch.load(f)

# ===== dive into torch.nn ======
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
# example:

torch.set_printoptions(edgeitems=2, linewidth = 75)

# toy data
t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c).unsqueeze(1) # <1>
t_u = torch.tensor(t_u).unsqueeze(1) # <1>

t_u.shape
#
n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)

train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_indices, val_indices
#
t_u_train = t_u[train_indices]
t_c_train = t_c[train_indices]

t_u_val = t_u[val_indices]
t_c_val = t_c[val_indices]

t_un_train = 0.1 * t_u_train
t_un_val = 0.1 * t_u_val

# linear model
# nn.Linear(in_features, out_features, bias = True)
linear_model = nn.Linear(1, 1)
linear_model(t_un_val)

linear_model.weight
linear_model.bias

# batching inputs
x = torch.ones(10, 1)
linear_model(x)

# constructing nn
linear_model = nn.Linear(1, 1)
# constructing
optimizer = optim.SGD(linear_model.parameters(), lr= 1e-2)

# you cannot see the parameter with this line alone
linear_model.parameters()
# use this instead
list(linear_model.parameters())

# define training loop
def training_loop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val, t_c_train, t_c_val):

    for epoch in range(1, n_epochs + 1):
        t_p_train = model(t_u_train)
        loss_train = loss_fn(t_p_train, t_c_train)

        t_p_val = model(t_u_val)
        loss_val = loss_fn(t_p_val, t_c_val)

        # optimizer has already been configured with linear_model
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if epoch == 1 or epoch % 1000 == 0:
            print(f'Epoch {epoch}, Training loss {loss_train.item():.4f},'f'Validation loss {loss_val.item():.4f}')


# constructing nn
linear_model = nn.Linear(1, 1)
# constructing
optimizer = optim.SGD(linear_model.parameters(), lr= 1e-2)
'''
linear_model and optimizer are both instantiated
'''

training_loop(
    n_epochs = 3000,
    optimizer = optimizer,
    model = linear_model,
    loss_fn = nn.MSELoss(),
    t_u_train = t_un_train,
    t_u_val = t_un_val,
    t_c_train = t_c_train,
    t_c_val = t_c_val
)

print(linear_model.weight)
print(linear_model.bias)

# replacing linear class to actual nn module

seq_model = nn.Sequential(nn.Linear(1,13),
                          nn.Tanh(),
                          nn.Linear(13,1))
seq_model

# alternatively
[param.shape for param in seq_model.parameters()]

# named parameters:
for name, param in seq_model.named_parameters():
    print(name, param.shape)
'''
0.weight torch.Size([13, 1])
0.bias torch.Size([13])
2.weight torch.Size([1, 13])
2.bias torch.Size([1])

为什么第一层nn.Linear(1,13)的权重的size是 (13,1)?:
nn.Linear(1,13)的意思是in_features = 1, out_features = 13，也就是他要把一个(n,1)的输入向量变成 (n, 13)输出，第一个维度n是batchsize，，用向量的语言描述：x@W^T -> (n, 1) @ (1, 13) = (n, 13), 故 W^T = (1, 13), 故W = (13, 1)
'''

# name with our specification
from collections import OrderedDict

seq_model = nn.Sequential(OrderedDict(
    [('hidden_linear', nn.Linear(1,8)),
     ('hidden_activation', nn.Tanh()),
     ('output_linear', nn.Linear(8,1))]))

seq_model
# access param with sub-module as attributes
seq_model.hidden_linear.bias
'''
Sequential(
  (hidden_linear): Linear(in_features=1, out_features=8, bias=True)
  (hidden_activation): Tanh()
  (output_linear): Linear(in_features=8, out_features=1, bias=True)
)
'''
# more explanatory name:
for name, param in seq_model.named_parameters():
    print(name, param.shape)
'''
hidden_linear.weight torch.Size([8, 1])
hidden_linear.bias torch.Size([8])
output_linear.weight torch.Size([1, 8])
output_linear.bias torch.Size([1])
'''

# inspecting parameters/ grads
optimizer = optim.SGD(seq_model.parameters(), lr= 1e-3)
'''
linear_model and optimizer are both instantiated
'''

training_loop(
    n_epochs = 5000,
    optimizer = optimizer,
    model = seq_model,
    loss_fn = nn.MSELoss(),
    t_u_train = t_un_train,
    t_u_val = t_un_val,
    t_c_train = t_c_train,
    t_c_val = t_c_val
)

print('output:', seq_model(t_un_val))
print('answer:', t_c_val)
# access grad by named sub-module
print('hidden:', seq_model.hidden_linear.weight.grad)

# comparing to linear model

from matplotlib import pyplot as plt

t_range = torch.arange(20., 90.).unsqueeze(1)
fig = plt.figure(dpi=600)

plt.xlabel('Fahrenheit')
plt.ylabel('Celsius')

# actual data: circles
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
# if you don't use detach, errors: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead
# model predicted data:
plt.plot(t_u.numpy(), seq_model(0.1*t_u).detach().numpy(), 'kx')

# model curve
plt.plot(t_range.numpy(), seq_model(0.1*t_range).detach().numpy(), 'c-')








