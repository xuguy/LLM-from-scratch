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