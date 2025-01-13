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