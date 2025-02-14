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
