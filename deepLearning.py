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
