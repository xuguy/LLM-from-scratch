# ======= 实现variable 类 =======
class Variable:
    def __init__(self, data):
        self.data = data

import numpy as np
data = np.array(1.0)
x = Variable(data)
print(x.data)

# ==============================

# ======= 实现 Function 类 =====
class Function:
    def __call__(self, input):
        x = input.data
        y = x**2
        output = Variable(y) # 装箱输出
        return output


f = Function()
x = Variable(np.array(2))
y = f(x)
print(type(y))
print(y.data)

# ===== redesign Function class ======
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError('method should be implemeted via hiarachi')

f = Function()
f.forward(x)

# hierachi
class Square(Function):
    def forward(self, x):
        return x**2

f = Square()
y = f(x)
y.data

# consecutive apply of function
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
A = Square()
B = Exp()
C = Square()
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)

# numerical grad
def numerical_diff(f, x, eps = 1e-4):

    # remember that actual variable stored in Variable.data
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data)/(2*eps)

f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)
type(dy)

# compound function numerical grad
def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print(dy)

# extend Variable class for backprop
# add self.grad to store grad
#%%
import numpy as np
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input
        return output
    def forward(self, x):
        raise NotImplementedError('implement forward by hierachi')
    def backward(self, gy):
        raise NotImplementedError('implement backward by hierachi')

class Square(Function):
    def forward(self, x):
        # 注意，把变量套上Variable class的操作在Function种实现了
        y = x**2
        return y
    def backward(self, gy):
        x = self.input.data
        gx = 2*x*gy
        return gx
    
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x)*gy
        return gx    
    
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# derive grad manually
#dy/dy
# y.grad = np.array(1.0)
# b.grad = C.backward(y.grad)
# a.grad= B.backward(b.grad)
# x.grad = A.backward(a.grad)
# print(x.grad)

#%%
# autograd
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self) # 设定（注明）y的creator
        self.input = input # 设定input
        self.output = output
        return output
    
# class Square(Function):
#     def forward(self, x):
#         # 注意，把变量套上Variable class的操作在Function种实现了
#         y = x**2
#         return y
#     def backward(self, gy):
#         x = self.input.data
#         gx = 2*x*gy
#         return gx
    
# class Exp(Function):
#     def forward(self, x):
#         y = np.exp(x)
#         return y
#     def backward(self, gy):
#         x = self.input.data
#         gx = np.exp(x)*gy
#         return gx    
    
# test
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# Function 通过 creator 和input记录连接信息
# Variable只需要记录creator
assert y.creator == C
assert y.creator.input ==b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

# backward with autograd
# b -> C -> y
y.grad = np.array(1.0)
C = y.creator
b = C.input
b.grad = C.backward(y.grad)
print(b.grad)

B = b.creator
a = B.input
a.grad = B.backward(b.grad)

A = a.creator
x = A.input
x.grad = A.backward(a.grad)
print(x.grad)

# extend for .backward() method
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        #只有被creator创建的Variable，才能微分
        f = self.creator
        if f is not None:
            x = f.input # f.input是x前面的一个Variable变量
            x.grad = f.backward(self.grad) #self.grad = gy
            x.backward() #递归调用：一直往前传

# test
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
y.backward() # 我们只需要调用一次backward，反向传播就会自动（递归地）进行
print(x.grad)
