# 阶跃函数实现 page 43
import numpy as np
import matplotlib.pylab as plt
def step_function(x):
    y=x>0
    return y.astype(np.int)

# sigmoid 函数实现
def sigmoid(x):
    return 1/(1+np.exp(-x))

x=np.array([-1.0, 1.0, 2.0])
sigmoid(x)

x=np.arange(-5.0, 5.0, 0.1)
y=sigmoid(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# relu 函数的实现
def relu(x):
    return np.maximum(0,x)

# page 59的第0层到第1层前向传播实现：
X=np.array([1.0, 0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])

print(W1.shape)
A1 = np.dot(X,W1)+B1
Z1 = sigmoid(A1)
print(f'A1:{A1}\nZ1:{Z1}')

# 第1层到第二层：
W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2 = np.array([0.1,0.2])

# 上一层的输出Z1变成了这一层的输入
A2 = np.dot(Z1,W2)+B2
Z2 = sigmoid(A2)

# 第2层到输出层
def identity_function(x):
    return x

W3 = np.array([[0.1,0.3],[0.2,0.4]])
B3 = np.array([0.1,0.2])
A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)

#往后的传播都是一样的，合并为以下：
def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network, x)
print(y)

# 实现softmax函数并使用softmax函数担任输出层的激活函数
# 这个函数经过改进，可以一定程度上避免溢出问题
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a

    return y

softmax(np.array([1010,1000,990]))

# ========== load MNIST dataset ==============
# # load MNIST data set
import sys,os, pickle

mnistPath = os.getcwd()+'\\DL-code'
sys.path.append(mnistPath)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=True)
print(f'x_train.shape: {x_train.shape}\nt_train.shape: {t_train.shape}')

# ===========================================

# ===== following code just for test, no need to run everytime =========================================
# from PIL import Image

# def img_show(img):
#     pil_img = Image.fromarray(np.uint8(img))
#     pil_img.show()

# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# img = x_train[0]
# label = t_train[0]
# print(label)  # 5

# print(img.shape)  # (784,)
# img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
# print(img.shape)  # (28, 28)

# img_show(img)
# ===============================================

# inference
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open(os.getcwd()+'\\DL-code\\ch03\\sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    # W1.shape = (784, 50)
    # b1.shape = (50,)
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print('Accuracy:' +str(float(accuracy_cnt)/len(x)))

# batch process
x, t = get_data()
network = init_network()

batch_size=100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch) # shape=(100,10)
    p = np.argmax(y_batch, axis=1) # 取每一行的最大值的index
    accuracy_cnt += np.sum(p==t[i:i+batch_size]) # t: truelabel

print(f'accuracy: {accuracy_cnt/len(x)}')


#============ chapter 4 ================
# test data (wasted)
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
t1 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # one-hot representation
t2 = 2

# contruct mean-squre loss function
def mean_squared_error(y, t):
    return 0.5*np.sum((y-5)**2)

# construct cross-entropy loss
def cross_entropy(y,t):
    delta = 1e-7 # to protect from val overflow
    return -np.sum(t*np.log(y+delta))


# mini-batch training

# load MNIST data set
import sys,os, pickle

mnistPath = os.getcwd()+'\\DL-code'
sys.path.append(mnistPath)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=True)
print(f'x_train.shape: {x_train.shape}\nt_train.shape: {t_train.shape}')

# random choose batch
train_size = x_train.shape[0]
batch_size= 10
batch_mask = np.random.choice(train_size, batch_size) # choose integer

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# cross-ent minibatch loss (one-hot)
def cross_entropy_error_1h(y,t):
    if y.ndim ==1:
        t = t.reshape(1,t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]

    return -np.sum(t*np.log(y+1e-7))/batch_size

# test if it works
cross_entropy_error_1h(np.array(y1),np.array(t1))

# cross-ent minibatch loss (non-onehot)
def cross_entropy_error_n1h(y,t):
    if y.ndim ==1:
        t = t.reshape(1,t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t]+1e-7))/batch_size

cross_entropy_error_n1h(np.array(y1),np.array(t2))

# differentiatial
def function_1(x):
    return 0.01*x**2+0.1*x

def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)

x = np.arange(0.0,20.0,0.1)
y = function_1(x)

numerical_diff(function_1, 5)

# partial derivatives
def function_2(x):
    return x[0]**2+x[1]**2

# numerical gradient
def numerical_gradient(f,x):
    h=1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        #cal f(x+h)
        x[idx] = tmp_val+h
        fxh1=f(x)

        #cal f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1-fxh2) / (2*h)

    return grad

numerical_gradient(function_2, np.array([3.0,4.0])) # (6.,8.)

# gradient method
def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    
    x=init_x.copy()
    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -= lr*grad
    
    return x
init_x = np.array([-3.0, 4.0])
tmp = gradient_descent(function_2, init_x = init_x, lr=0.1, step_num=100)

function_2(tmp)

# simple net 
import sys
mnistPath = os.getcwd()+'\\DL-code' # pardir
sys.path.append(mnistPath)

from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient
import numpy as np

class simpleNet:
    def __init__(self):
        self.W = np.random.rand(2,3)# 用高斯分布初始化权重

    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y=softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
np.argmax(p) # predict label
t =np.array([0,0,1]) # true label
net.loss(x,t)

# cal gradients
# def
def f(w):
    return net.loss(x, t)
# or use lambda : f = lambda w: net.loss(x, t)
f = lambda w: net.loss(x, t) # w is a fake input var, because within numerical_gradient, f must take at least 1 input
dW = numerical_gradient(f, net.W)
print(dW)

# this is how the imported numerical_gradient function iter through all element of weight matrix
it = np.nditer(net.W, flags=['multi_index'], op_flags=['readwrite'])
while not it.finished:
    idx = it.multi_index
    print(idx)
    it.iternext()

#%%
# 实现一个2层神经网络（隐藏层数为1层）识别手写数字：
# avoid appending pardir multiple times
import sys, os
mnistPath = os.getcwd()+'\\DL-code' # pardir
if not sum(['DL-code' in i for i in sys.path]):
    print('curPath does not have required path, imported else where')
    sys.path.append(mnistPath)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):

        self.params = {}
        self.params['W1'] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std*np.random.rand(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    def loss(self, x, t):
        y=self.predict(x)

        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1) # 每行的最大值的index
        t = np.argmax(t, axis = 1) # 找出为1的index

        accuracy = np.sum(y==t)/float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        # 这里需要好好解释一下传入的self.params['W1']系列参数究竟有什么用:
        # loss_W 不需要接受传入参数进行计算，也就是loss_W(self.params['W1']) 和loss_W(any_value)是一样的，但是在numerical_gradient(f, x)的内部会计算f(x+h)和f(x-h)，这里self.params['W1']被当作x传入numerical_gradient，loss_W被当作f传入，而loss_W由self.params['W1']间接决定（参照gradient.py中的具体实现方法）
        return grads


# initialize
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params['W1'].shape)

# test with generated data
x = np.random.rand(100, 784)
t = np.random.rand(100, 10)
y = net.predict(x)

grads = net.numerical_gradient(x, t)

#%%
# mini-batch version
import numpy as np
mnistPath = os.getcwd()+'\\DL-code' # pardir
if not sum(['DL-code' in i for i in sys.path]):
    print('curPath does not have required path, imported else where')
    sys.path.append(mnistPath)
from dataset.mnist import load_mnist
# use before defined TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label = True)

train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size/batch_size, 1)

# set params
iters_num = 10#use 10 iters first to see how many time used, then 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size/batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):

    #get mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -=learning_rate*grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    

    if i%iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('train acc, test acc | '+str(train_acc) + ', ' + str(test_acc))

        



# 这里由于学习率以及batch选择的随机性问题，在前10次（甚至前100次）循环中loss下降的都不大，甚至看不出下降的趋势，程序本身并没有问题


# %% 
#================ backward propogation =====================

# multiply layer
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x*y

        return out
    
    def backward(self, dout):
        # dout be the received partial derivative
        dx = dout*self.y
        dy = dout*self.x

        return dx, dy
    

# test
# input: all the elements that will affect the price of the apple
apple = 100
apple_num=2
tax = 1.1

# layer: 2 steps: get the number of apple, then consider tax
# a layer = a multiply node
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)

# backward:
# the whole price (consider number of apple and tax)
dprice = 1

# how will the other variable change
# each mul node has 2 input and output 1 value
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)


# add layer, similar to mul layer

class AddLayer:

    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout*1
        dy = dout*1

        return dx, dy
    

# test with a complicated network

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# initialize, from graph (page 138) we can see that there are 3 mul nodes and 1 add node
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

#forward
# 1st layer
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)

# 2nd layer
all_price = add_apple_orange_layer.forward(apple_price, orange_price)

# 3rd layer
price = mul_tax_layer.forward(all_price, tax)

#backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(f'price:{price}') # should be 715

print(f'dapple_num:{dapple_num}, dapple:{dapple}, dorange:{dorange}, dorange_num:{dorange_num}, dtax:{dtax}')

# it is crital how each layer is organize, they are ordered
# each initialzed layer is a node, all data are saved within the instance of the class
# 为什么要用类来实现层（节点）：因为类可以保存计算过的数据

# 激活层函数的实现
import numpy as np
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0

        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

# sigmoid layer (Page 144) please revisit how this layer is constructed
class Sigmoid:
    def __inti__(self):
        self.out = None
    
    def forward(self, x):
        out = 1/(1+np.exp(-x))
        self.out = out

        return out
    
    def backward(self, dout):
        dx = dout*(1.0 - self.out)*self.out

        return dx

# Affine层的实现
import numpy as np

X = np.random.rand(2)
W = np.random.rand(2,3)
B = np.random.rand(3)

# Y = X@W+B : (1,3)
# parL/parX = (parL/parY)@W^T
# parL/parW = X^T@(parL/parY)
# 注意 一维向量(R,)既可以左乘（作为行向量），又可以右乘（作为列向量）
# 参考page 147/148的shape及传播图
# batch版本的计算和单向量版本的计算在数学标识上一样，在coding里也一样

class Affine:
    
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out
    
    def backward(self, dout):
        # 为什么输出dx：notation problem：正向传播Y=XW+B，反向传播：dY = ？dX，这里的dx就是这个意思，不要想多了
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        # self.db为什么要这样算（看page148图 parL/parB)
        self.db = np.sum(dout, axis = 0)

        return dx


# softmax layer
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        # x input, t label
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
    
    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t)/batch_size

        return dx




# %%
# 2 layer network, a more efficient realization
import numpy as np
mnistPath = os.getcwd()+'\\DL-code' # pardir
if not sum(['DL-code' in i for i in sys.path]):
    print('curPath does not have required path, imported else where')
    sys.path.append(mnistPath)
from dataset.mnist import load_mnist

# common.layers 里面的函数都是我们之前定义过的，跑通后测试一下之前定义的函数是否也能够跑通
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

# TwoLayerNet

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):

        #初始化权重：随机初始
        self.params = {}
        self.params['W1'] = weight_init_std*np.random.rand(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std*np.random.rand(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # define layer
        self.layers = OrderedDict() # what is this?
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()
    
    # 通过逐层调用self.layers(一个orderdDict对象)，逐层把输入值x往前传，直到最后一层，也即输出结果（predict）
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
    
    def loss(self, x, t):
        y = self.predict(x)# 获取输出结果
        return self.lastLayer.forward(y, t) # lastLayer即cross_ent层，这一层的forward就是在计算交叉熵损失
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1) # 获取每一行最大值的列下标
        # 下面这个if判断实际上可以兼容非one-hot形式的标签
        if t.ndim != 1: t = np.argmax(t, axis = 1) #onehot里为1的idx
        accuracy = np.sum(y == t)/float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        #用数值方法计算grad
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    
    def gradient(self, x, t):
        # 用反向传播方法计算grad
        # 下面这个前向传播调用self.loss，不会返回任何值，作用是计算后向传播（见下面）所需要的self.LastLayer里面的值：self.loss里面会执行一次lastLayer里面的的forward
        self.loss(x, t)
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        # 整个网络的前向传播顺序：X->Affine1->Relu1->Affine2->softmax->分类结果
        # 按顺序反转list，然后从后向前执行反向传播，layers就会保存每层的反向传播计算结果。
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

# gradient check:
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label = True)

network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

x_batch = x_train[:3]

t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 求各个权重的绝对误差的平均值（看两种计算grad的方法是否一致）
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))

    print(key + ':' + str(diff)) # 结果和书上不一样，因为网络权重初始化用的是随机初始化，我们并不知道书上用的随机种子是多少

#%%
# backprop in practice:

network = TwoLayerNet(input_size=784, hidden_size=50, output_size = 10)

iters_num = 10000 # backprop is fast, so its ok  
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size/batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate*grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f'train_acc: {train_acc:.3f}, test_acc: {test_acc:.3f}')

# =================== Ch 06 =======================
# ================= optimizaer ====================

import numpy as np
class Momentum:
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum # a.k.a. alpha (<1)
        self.v = None

    def update(self, params, grads):

        # v会以字典型变量的形式保存与参数结构相同的数据
        # 如果是第一次更新，那么把v的所有值全部初始化为0
        if self.v is None:
            self.v = {}
            # 回忆params的结构：key-value，其中value就是各层的参数矩阵
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] +=self.v[key]

class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            # h的结构和待更新参数一样
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            #非RMSPROP
            self.h[key] += grads[key]*grads[key]
            params[key] -= self.lr*grads[key]/(np.sqrt.h[key] + 1e-7)
