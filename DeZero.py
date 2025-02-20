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


# ============== almost full setup of DeZero
# 简化backward方法：省略手动输入np.array(1.0)
# 强制使用ndarray数据类型，避免error
#%%
import numpy as np
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


# ============ phase 2 =============
#%%
# 修改Function类以适应多个输入和输出

class Function:
    def __call__(self, *inputs):
        # 加入星号后，所有的inputs会被打包成一个元组输入
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        # 如果ys不是元组，就把它修改为元组,因为ys有可能是单个元素
        # 注意看新的forward方法，返回的是单个y，因此不一定是元组
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs)>1 else outputs[0]
    
    def forward(self,x):
        raise NotImplementedError('Function.forward not implemented')
    
    def backward(self, gy):
        raise NotImplementedError('Function.backward not implemented')
    
class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)
# test: 改进（适应任意个参数）前
# xs = [Variable(np.array(2)), Variable(np.array(3))]
# f = Add()
# ys = f(xs)
# y = ys[0]
# print(y.data)

# test：改进后
x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
f = Add()
y = f(x0, x1)
print(y.data)

# 改进Add的forward的方法：
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y # 返回一个元素而不需要返回元组
    
    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)

x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
y = add(x0, x1)
print(y.data)

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
            #gys是后一个变量的grad
            gys = [output.grad for output in f.outputs]
            # gxs是前一个变量的grad
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            # f.inputs是Variable，x.grad是数值
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    funcs.append(x.creator)



class Square(Function):
    def forward(self, x):
        y = x**2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2*x*gy
        return gx

def square(x):
    f = Square()
    return f(x)

x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

z = add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)

# test add 2 same number
x = Variable(np.array(3.0))
y = add(add(x, x),x)
y.backward()
print(f'y:{y.data}, x.grad: {x.grad}')
'''
上面这种计算grad的方式虽然可以避免无法正确计算使用同一个Variable作为参数的函数的grad的问题，但也会有新的问题：一旦我们需要重复使用同一个实例（为了节省内存，重复使用x=Variable()），之前计算的倒数会被加在新的导数上，而我们既然重复使用同一个实例那就必然希望该实例是全新的未使用过的状态，因此需要“重置导数”
'''
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
            #gys是后一个变量的grad
            gys = [output.grad for output in f.outputs]
            # gxs是前一个变量的grad
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            # f.inputs是Variable，x.grad是数值
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    funcs.append(x.creator)
    # 把self.grad设定为None就行
    def cleargrad(self):
        self.grad = None

x = Variable(np.array(3.0))
y = add(x,x)
y.backward()
print(x.grad) # 2.0

x.cleargrad()
y = add(add(x,x),x)
y.backward()
print(x.grad) # 3.0
'''
因此，在调用第二个计算的y.backward之前调用x.cleargrad()，就可以充值变量中保存的导数，这样就可以使用同一个变量实例来执行其他操作计算，从而达到节省内存的目的
'''
# %%
# step 15/16:复杂计算图的实现
# 增加“辈分”变量
import numpy as np
import weakref
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

# class Variable:
#     def __init__(self, data):
#         if data is not None:
#             if not isinstance(data, np.ndarray):
#                 raise TypeError('{} is not supported'.format(type(data)))
#         self.data = data
#         self.grad = None
#         self.creator = None
#         self.generation = 0
    
#     def set_creator(self, func):
#         self.creator = func
#         self.generation = func.generation + 1
#     def backward(self):
#         # grad的shape与数据类型与self.data（input）相同
#         if self.grad is None:
#             self.grad = np.ones_like(self.data)

#         funcs = [self.creator]
#         while funcs:
#             # print(funcs) # 列表每次循环都会删掉一个Function实例，并添加一个Function实例
#             f = funcs.pop()
#             #gys是后一个变量的grad
#             gys = [output.grad for output in f.outputs]
#             # gxs是前一个变量的grad
#             gxs = f.backward(*gys)
#             if not isinstance(gxs, tuple):
#                 gxs = (gxs,)
#             # f.inputs是Variable，x.grad是数值
#             for x, gx in zip(f.inputs, gxs):
#                 if x.grad is None:
#                     x.grad = gx
#                 else:
#                     x.grad = x.grad + gx

#                 if x.creator is not None:
#                     funcs.append(x.creator)
#     # 把self.grad设定为None就行
#     def cleargrad(self):
#         self.grad = None

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        # func的generation就是inputs的gen中最大的那个
        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs 
        # 注意观察上面的outputs是如何生成的，就可以理解下面这里的列表推导式
        self.outputs = [weakref.ref(output) for output in outputs]# original: self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self,x):
        raise NotImplementedError('Function.forward not implemented')
    
    def backward(self, gy):
        raise NotImplementedError('Function.backward not implemented')

# # test
# generations = [2, 0, 1, 4, 2]
# funcs = []
# for g in generations:
#     f = Function()
#     f.generation = g
#     funcs.append(f)
# [f.generation for f in funcs]

# funcs.sort(key=lambda x: x.generation)
# [f.generation for f in funcs]
# f = funcs.pop()
# f.generation # should be 4
class Variable(object):
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
    
    def set_creator(self, func):
        self.creator = func
        # 在正向传播(__call__)的过程中对变量对output设定generation，为什么只对output？：因为只有output会被调用set_creator
        self.generation = func.generation + 1


    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = []
        # 用于防止同一个函数被多次添加到funcs中，从而防止一个函数的backward方法被错误地多次调用: 图论有关的算法常用技巧，用来防止cycle
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            # gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)
                    # add_func加入的是当前函数的输入变量的creator：例如，funcs=[B,C],那么C会被pop出作为当前函数，C的输入变量是a，a的creator：A 将会被加入funcs
                    # 最精髓的就是这个add_func函数：每次添加输入变量的creator后，funcs列表都会被重新排序，以保证funcs中可能存在的多个输入变量的creator的generation较大的排在后面，pop被优先取出
    def cleargrad(self):
        self.grad = None

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y # 返回一个元素而不需要返回元组
    
    def backward(self, gy):
        return gy, gy


class Square(Function):
    def forward(self, x):
        y = x**2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2*x*gy
        return gx
    
def add(x0, x1):
    return Add()(x0, x1)

def square(x):
    f = Square()
    return f(x)

x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()
print(y.data) # 32
print(x.grad) # 64


# %%
# use weak refernce: to avoid loop ref, save memory
import weakref
a = np.array([1,2,3])
b = weakref.ref(a) # <weakref at 0x000001A5F4A78220; to 'numpy.ndarray' at 0x000001A5F4A9DD10>
b() # array([1, 2, 3])
c = a
c # array([1, 2, 3])
c() # not callable

# 利用弱引用改造上面的class定义
import numpy as np
import weakref

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        # func的generation就是inputs的gen中最大的那个
        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs 
        # 注意观察上面的outputs是如何生成的，就可以理解下面这里的列表推导式
        # 函数输出变量这一环节使用弱引用，打破循环应用
        self.outputs = [weakref.ref(output) for output in outputs]# original: self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self,x):
        raise NotImplementedError('Function.forward not implemented')
    
    def backward(self, gy):
        raise NotImplementedError('Function.backward not implemented')

class Variable(object):
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
    
    def set_creator(self, func):
        self.creator = func
        # 在正向传播(__call__)的过程中对变量对output设定generation，为什么只对output？：因为只有output会被调用set_creator
        self.generation = func.generation + 1

    def backward(self, retain_grad = False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = []
        # 用于防止同一个函数被多次添加到funcs中，从而防止一个函数的backward方法被错误地多次调用: 图论有关的算法常用技巧，用来防止cycle
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            # gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)
            # retain grad adaptation：在backwards计算完grad以后，删除输出的grad（y.grad）,保留x.grad继续反向传播
            if not retain_grad:
                for y in f.outputs:
                    # 注意，f.outputs 是弱引用，因为f.outputs就是Function类里面的self.outputs，which已经被改造成弱引用
                    y().grad = None

    def cleargrad(self):
        self.grad = None

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y # 返回一个元素而不需要返回元组
    
    def backward(self, gy):
        return gy, gy


class Square(Function):
    def forward(self, x):
        y = x**2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2*x*gy
        return gx
    
def add(x0, x1):
    return Add()(x0, x1)

def square(x):
    f = Square()
    return f(x)

# test#1
for i in range(10):
    x = Variable(np.random.rand(10000))
    y = square(square(square(x)))

# test#2: retain_grad adaptation test
x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
t = add(x0, x1)
y = add(x0, t)
y.backward()

print(f'y.grad: {y.grad}\nt.grad: {t.grad}\nx0.grad: {x0.grad}\nx1.grad: {x1.grad}')

'''
正反向传播的变量之间的联系是如何建立的呢？换句话说，前一个变量的outputs是如何与下一个变量的inputs建立联系的呢？
注意观察Variable里面的backward方法里面的这一小段：
while funcs:
f = funcs.pop()
# gys = [output.grad for output in f.outputs]
gys = [output().grad for output in f.outputs]
gxs = f.backward(*gys)
if not isinstance(gxs, tuple):
gxs = (gxs,)
for x, gx in zip(f.inputs, gxs):
if x.grad is None:
    x.grad = gx
else:
    x.grad = x.grad + gx
if x.creator is not None:
    add_func(x.creator)
#========== code end =============
# keypoint：
f.outputs->gys->gxs; for x, gx in zip(f.inputs, gxs)
'''

# 用Config类切换正向传播模式/反向传播模式：省去不必要的计算
import numpy as np
import weakref

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Config:
    enable_backprop = True

class Function:
    def __call__(self, *inputs):
        # forward pass
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        if Config.enable_backprop:
        # ======== BP code ==========
            # func的generation就是inputs的gen中最大的那个
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self) # 设置前后连接
            self.inputs = inputs 
            # 注意观察上面的outputs是如何生成的，就可以理解下面这里的列表推导式
            # 函数输出变量这一环节使用弱引用，打破循环应用
            self.outputs = [weakref.ref(output) for output in outputs]# original: self.outputs = outputs
        # ========= BP code end ==========
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self,x):
        raise NotImplementedError('Function.forward not implemented')
    
    def backward(self, gy):
        raise NotImplementedError('Function.backward not implemented')

# 重新运行子类定义，确定父类的修改被应用
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y # 返回一个元素而不需要返回元组
    
    def backward(self, gy):
        return gy, gy


class Square(Function):
    def forward(self, x):
        y = x**2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2*x*gy
        return gx
    
def add(x0, x1):
    return Add()(x0, x1)

def square(x):
    f = Square()
    return f(x)

# test
Config.enable_backprop = True
x = Variable(np.ones((100,100,100)))
y = square(square(square(x)))
y.backward()
print(x.grad)

Config.enable_backprop = False
x = Variable(np.ones((100, 100, 100)))
y = square(square(square(x)))
y.backward() # error
print(x.grad) # still None

# 使用with语句处理需要临时修改config的情况



# %%
# ====== step 19 ======让变量变得更好用：

import numpy as np

class Variable(object):
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
    
    def set_creator(self, func):
        self.creator = func
        # 在正向传播(__call__)的过程中对变量对output设定generation，为什么只对output？：因为只有output会被调用set_creator
        self.generation = func.generation + 1

    def backward(self, retain_grad = False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = []
        # 用于防止同一个函数被多次添加到funcs中，从而防止一个函数的backward方法被错误地多次调用: 图论有关的算法常用技巧，用来防止cycle
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            # gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)
            # retain grad adaptation：在backwards计算完grad以后，删除输出的grad（y.grad）,保留x.grad继续反向传播
            if not retain_grad:
                for y in f.outputs:
                    # 注意，f.outputs 是弱引用，因为f.outputs就是Function类里面的self.outputs，which已经被改造成弱引用
                    y().grad = None

    def cleargrad(self):
        self.grad = None

    # 新增shape\ndim\size等方法，让Variable实例和ndarray实例一样好用
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)# 返回data dim=0 的元素数量

    # 重写__repr__方法，自定义print(Variable)输出的内容
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        
        p = str(self.data).replace('\n', '\n' + ' '*9)
        return 'variable(' + p +')'
    
    # 重载 * 运算符
    def __mul__(self, other):
        # 在使用 * 计算Variable类实例时，调用的就是__mul__方法，此时，运算符 * 左侧的a作为self参数，右侧的b作为other参数传给了__mul__方法，b在 * 的右侧，调用的特殊方法是__rmul__
        return mul(self, other)
    # 重载 + 运算符
    def __add__(self, other):
        return add(self, other)

    
# test
x = Variable(np.array([[1,2,3], [4,5,6]]))
len(x) # len() 自动调用Variable实例的 __len__方法

print(x)

# =========== step 20 ============
# 运算符重载

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y # 返回一个元素而不需要返回元组
    
    def backward(self, gy):
        return gy, gy

class Square(Function):
    def forward(self, x):
        y = x**2
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2*x*gy
        return gx

class Mul(Function):
    def forward(self, x0, x1):
        y = x0*x1
        return y
    def backward(self, gy):
        # 因为Mul继承了Function类，因此也继承了Function类的inputs/data
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy*x1, gy*x0

# make it easy to use 
def add(x0, x1):
    return Add()(x0, x1)

def square(x):
    f = Square()
    return f(x)
    

def mul(x0, x1):
    return Mul()(x0, x1)

# test
a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

y = add(mul(a, b), c)

y.backward()

print(y, a.grad, b.grad)

# 我们还是觉得 add(mul(a, b), c) 这种写法太麻烦，有没有办法可以直接用 a*b+c这种自然的表达方法呢？
y = a*b+c
y.backward()
print(a.grad, b.grad)
'''
注意，重载运载符以后要重新运行Function, Add，Square, Mul 的定义以及他们各自的该作过的易用函数版本
'''

