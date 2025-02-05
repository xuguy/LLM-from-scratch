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

'''
之前遇到过的一个AssertionErrot的问题在于：子类继承父类后，如果再改写父类，新的父类不会直接被应用在子类的方法中。因此，你需要重新运行一遍定义子类的代码，这样新的父类才会被子类正确继承。
例子：
class Parent:  
    def greet(self):  
        print("Hello from Parent!2")  
  
class Child(Parent):  
    def greet(self):  
        # 调用父类的greet方法  
        super().greet()  
        # 打印子类的消息  
        print("Hello from Child!") 
tmp = Child()
tmp.greet()
'''



    
class Square(Function):
    # 函数中的forward方法会覆盖父类Function中的forward
    '''
    例子：
    class par:
    def __call__(self):
        self.woof()
        print('123')

    class chil(par):
        # 实际上，子类实例运行的是父类中的__call__方法，只不过父类中__call__方法中的某个函数（woof）被子类覆写了
        def woof(self):
            print('555')
    tmp = chil()()
    '''
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
            x.backward() #递归调用：一直往前传，直到遇见一个self.creator函数为None的变量，也就是第一个输入变量

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

#%%
#使用循环实现backward
class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        #在进行一次完整的正向传播后，每一个变量的creator都会被设定
        self.creator = None


    def set_creator(self, func):
        self.creator = func

    def backward(self):
        funcs = [self.creator]
        while funcs:
            # print(funcs) # 列表每次循环都会删掉一个Function实例，并添加一个Function实例
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)
                
# test
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)
'''
使用递归和循环实现的反向传播本质上是一样的，正向传播进行完毕后，需要指定一个反向传播的起始节点并手动给出一个输入的初始值，也即输入一个grad，一般使用y.grad = np.array(1.0)。
'''

# make it easier to use (做一些变量类型的规定以及容错、报错)

# 把函数写成类的用法比较麻烦，下面自动化这一过程
def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

# test
x = Variable(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b) # or: y = square(exp(square(x)))

y.grad = np.array(1.0)
y.backward()
print(x.grad)

# 简化backward方法：省略手动输入np.array(1.0)
# 强制使用ndarray数据类型，避免error
#%%
class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'input type:{type(data)} is not supported')
        self.data = data
        self.grad = None
        #在进行一次完整的正向传播后，每一个变量的creator都会被设定
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        # grad的shape与数据类型与self.data（input）相同
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            # print(funcs) # 列表每次循环都会删掉一个Function实例，并添加一个Function实例
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

# test:
# x = Variable(1.0) # raise error:
# x = Variable(None) # OK
'''
# 如何处理0维数据：
x = np.array(1.0)
y = x**2
print(type(x), x.ndim) # <class 'numpy.ndarray'> 0
print(type(y)) # <class 'numpy.float64'> : 不是ndarray！
'''


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

# 重写Function类：
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

class Square(Function):
    # 函数中的forward方法会覆盖父类Function中的forward
    '''
    例子：
    class par:
    def __call__(self):
        self.woof()
        print('123')

    class chil(par):
        # 实际上，子类实例运行的是父类中的__call__方法，只不过父类中__call__方法中的某个函数（woof）被子类覆写了
        def woof(self):
            print('555')
    tmp = chil()()
    '''
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
    
def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

x = Variable(np.array(0.5))

y = square(exp(square(x)))
y.backward()
print(x.grad)

# ======= finish step 09 ===========




