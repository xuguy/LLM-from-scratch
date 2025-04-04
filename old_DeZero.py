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
import weakref
import contextlib

class Config:
    enable_backprop = True



@contextlib.contextmanager
def using_config(name, value):

    # get the value of Config.name
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

'''
with using_config('enable_backprop', False):
    x = Variable(np.array(2.0))
    y = square(x)
'''
# make it easier to use
def no_grad():
    return using_config('enable_backprop', False)

with no_grad():
    x = Variable(np.array(2.0))
    y = square(x)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x
# step 21, 运算符重载：为了让Variable能供兼容ndarray的计算
def as_variable(obj):
    '''
    如果 obj 是Variable 实例， 则不做任何修改 直拨返。否则，将只转换为Variable实例
    '''
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Function:
    def __call__(self, *inputs):
        #step21 reload operator
        inputs = [as_variable(x) for x in inputs]

        # forward pass
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        if Config.enable_backprop:
        # ======== BP code ==========
            # func's generation defined by the maximum self.gen of all inputs
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self) # link input/output
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

class Variable(object):
    # step21：运算符重载，调高实例调用运算符优先级，高者优先调用
    __array_priority__ = 10086
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
    # 运算符左右对称性改造
    def __rmul__(self, other):
        return mul(self, other)
    # 重载 + 运算符
    def __add__(self, other):
        return add(self, other)
    
    def __radd__(self, other):
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
    # step21 运算符重载：与float和int一起使用
    x1 = as_array(x1)

    return Add()(x0, x1)


def square(x):
    f = Square()
    return f(x)
    

def mul(x0, x1):
    x1 = as_array(x1)
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

# 测试Variable与ndarray的兼容性
x = Variable(np.array(2.0))
y = x + np.array(3.0)

print(y)

# step21 运算符重载：与float和int一起使用，改造方法见add()函数定义
# test
x = Variable(np.array(2.0))
y = x+3.0
print(y)

# 运算符左右对称性兼容：见Variable内的__radd__/__rmul__
x = Variable(np.array(2.0))
y = 1.0+3.0*x

# step21: 左项为ndarray实例的情况
# 把实例魔法参数设高 ：__array_priotity__ = 10086，这样优先级较高的实例的相应运算符方法会被优先调用
tmp = np.array(2.0)
tmp.__array_priority__ # 0.0

x = Variable(np.array(1.0))
y = np.array([2.0]) + x

# 更多运算符重载的实现
'''
__neg__ sub rsub truediv rtruediv pow
'''

class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy
    
def neg(x):
    return Neg()(x)

Variable.__neg__ = neg # 这样就不需要在Variable定义里面加__neg__

# test
x = Variable(np.array(2.0))
y = -x
print(y)

class Sub(Function):
    def forward(self, x0, x1):
        y = x0-x1
        return y
    def backward(self, gy):
        return gy, -gy
    
def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0) # 交换x1和x0

Variable.__sub__ = sub
Variable.__rsub__ = rsub

x = Variable(np.array(2.0))
y1 = 2.0 - x
y2 = x - 1.0
print(y1, y2)

class Div(Function):
    def forward(self, x0, x1):
        y = x0/x1
        return y
    def backward(self, gy):
        # self.inputs来源: Div()(x0, x1)
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy/x1
        gx1 = gy*(-x0/x1**2)
        return gx0, gx1
    
def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv

x = Variable(np.array(2.0))
y1= 1.0/x
y2 = x/1.0
print(y1, y2)

class Pow(Function):
    def __init__(self, c):
        self.c = c
    def forward(self, x):
        y = x**self.c
        return y
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c*x**(c-1)*gy
        return gx
    
def pow(x, c):
    return Pow(c)(x)

Variable.__pow__ = pow

x = Variable(np.array(2.0))
y = x**3
print(y)
y.backward()
x.grad
# =========== draft ends here =============

# ====== step 23 packaging =======
# from now on, you no longer need to rerun those codes before, you can import everything
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
# from dezero import Variable
from dezero import * # is there any problem/drawback here by importing *?

# test if import successful
x = Variable(np.array(1.0))
y = 1. + x-1 # 我们已经定义了add/mul,但是没有定义sub和pow,这时候必须通过setup_variable()
y.backward()
print(y, x.grad)

# benchmark test for backprop
# sphere function
def sphere(x, y):
    z = x**2+y**2
    return z
x = Variable(np.array([1.0]))
y = Variable(np.array([1.0]))
z = sphere(x, y)
z.backward()
print(x.grad, y.grad)

# matyas function
def matyas(x, y):
    z = 0.26*(x**2 + y**2) - 0.48*x*y
    return z




def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2  - 14*y + 6*x*y + 3*y**2)) * (30 + (2*x- 3*y) **2 * (18- 32*x + 12*x**2 + 48*y-36*x*y + 27*y**2))
    return z

x = Variable(np.array([1.0]))
y = Variable(np.array([1.0]))
z = goldstein(x, y)
z.backward()
print(x.grad, y.grad)

# ===== visualize compute-graph
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero.utils import plot_dot_graph
from dezero import Variable
# from DL-dezero.core import Variable


# verify graphviz
def f(x):
    y = x**3
    return y

x = Variable(np.array([1.0]))
y = f(x)
# create_graph = True ->启动2次反向传播以计算高阶导数
y.backward(create_graph=True)
print(x.grad)

gx = x.grad
x.cleargrad()
gx.backward()
print(x.grad)


# test: try to cal higher order grad of sin/cos
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable, as_array
import dezero.functions as F
from dezero.utils import plot_dot_graph

x = Variable(np.array([1.0]))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

iters = 0 # 0=1阶导数，1=2阶导数

for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

# 绘制计算图
gx = x.grad
gx.name = 'gx' + str(iters+1)
plot_dot_graph(gx, verbose = False, to_file='tanh.png')

# example2: step36中的example：导数作为中间变量进入别的函数
x = Variable(np.array([2.5]))
y = x**2
y.backward(create_graph=False)
gx = x.grad
x.cleargrad()
# x.cleargrad()放在哪里都行，只要放在y.backward()后，以及z.backward()前，也即相邻的两次backward前，即可。
z = gx**3+y

# 如果我们在上面的y.backward(create_graph=False)，即在第一次反向传播的计算中中不创建计算图，那么第一次反向传播中gx = dy/dx = 2*x这部分运算就不会有计算图，也就是说，gx和x之间的连接不存在，因此在下面z.backward()，尽管z=gx**3+y，但是gx和x没有联系，因此x.grad=0+dy/dx=2*x
z.backward()
print(x.grad)

# test of multi-var
tmp = np.array([[1,2,3],[4,5,6]])
for i in tmp:
    print(i)

tmp2 = Variable(tmp)
z = tmp2**2
z.backward()
tmp2.grad
'''
关于多元函数以及向量、矩阵形式的计算与反向传播，目前还缺少针对矩阵运算的函数定义，因此暂时还无法处理矩阵形式的反向传播
'''
# test: try to cal higher order grad of sin/cos
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
from dezero import as_variable
import dezero.functions as F
# from dezero.utils import plot_dot_graph

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.reshape(x, (6,))
y.backward()
print(x.grad)
'''
variable([[1 1 1]
          [1 1 1]])
因为reshape不对输入进行任何树枝上的修改，仅变换形状，因此x.grad和x具有相同的形状，就像reshape从来没发生过一样
'''

# 测试Variable中的 reshape方法
# all make sense. remember, batched data always in the 0th dim of Variable
xs = Variable(np.random.randn(2,2,3))
x = np.random.randn(2,2,3)
y = xs.transpose((1,0,2))
y.backward()
xs.grad


# 测试step 39中实现的扩展版Sum

y = x.sum(keepdims=True)
y # which is a scalar with dims preserved
y.shape

# if keepdims=False by default, the resulted y will be a col vector, which is not desired


if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
from dezero import as_variable
import dezero.functions as F

# test code for sum() backward# 1
x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.sum(x, axis=0) #shape (3, )
y.backward()
print(y)
x.grad

# test code for sum() backward # 2
y = Variable(np.random.randn(3,5))
x = Variable(np.random.randn(2,3))
z = F.matmul(x,y)
z.backward()
x.grad
y.grad

#%%

# 简单的线性回归模型，用dezero实现：

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.dirname(__file__))
import numpy as np
from dezero import Variable
from dezero import as_variable
import dezero.functions as F

#%%
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2*x + np.random.rand(100, 1)
# x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1,1))) # (1,1)
b = Variable(np.zeros(1)) # (1,)

def predict(x):
    y = F.matmul(x,W) + b
    return y


lr = 0.1
iters = 100

for i in range(iters):

    # forward
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    # cleargrad before backward
    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr*W.grad.data
    b.data -= lr*b.grad.data

    print(W,b,loss)

#%%
# neural network simple:

import numpy as np
from dezero import Variable
import dezero.functions as F

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2*np.pi*x) + np.random.rand(100, 1)

# initialize weights
I, H, O = 1, 10, 1 # input hidden output layers
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))

W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))

def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid_simple(y)
    y = F.linear(y, W2, b2)
    return y

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)
    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr*W1.grad.data
    b1.data -= lr*b1.grad.data
    W2.data -= lr*W2.grad.data
    b2.data -= lr*b2.grad.data
    if i % 1000 == 0:
        print(loss)

# plot to check how well the nn fit to toy sin() data
import matplotlib.pyplot as plt
x_t = np.linspace(0,1,100).reshape(100,1)
y_pred = predict(x_t).data

# pretty well indeed
plt.plot(x_t, y_pred)

# layer class
import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero import Parameter
from dezero.layers import Layer



layer = Layer()
layer.p1 = Parameter(np.array(1))
layer.p2 = Parameter(np.array(2))
layer.p3 = Variable(np.array(3))
layer.p4 = 'test'

print(layer._params)
'''
out:
{'p2', 'p1'}
'''

for name in layer._params:
    print(name, layer.__dict__[name])
'''
out:
{'p2', 'p1'}
p2 variable(2)
p1 variable(1)
'''

# 使用Linear类实现神经网络：
#%%
import numpy as np
from dezero import Variable, Layer
from dezero.models import Model
import dezero.functions as F
import dezero.layers as L


# toy data
np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2*np.pi*x) + np.random.rand(100,1)

# (old)assign out_size to Linear layer
# l1 = L.Linear(10)
# l2 = L.Linear(1)

# 将Layer作为参数加入Layer的_params中后，我们就可以把作为Layer的Linear实例（L.Linear)传入Layer的实例model的_params中统一管理
model = Layer()
model.l1 = L.Linear(5)
model.l2 = L.Linear(3)

# 定义网络结构：
def predict(model, x):
    # 所有的参数都在model中统一管理
    y = model.l1(x)
    y = F.sigmoid_simple(y)
    y = model.l2(y)
    return y

# 测试访问所有参数并重置所有参数的梯度：
for p in model.params():
    #由于params()是yield构造的生成器，因此需要逐个访问
    print(p)

model.cleargrads()

lr = 0.2
iters = 1000

for i in range(iters):
    y_pred = predict(model, x)
    loss = F.mean_squared_error(y, y_pred)

    # l1.cleargrads()
    # l2.cleargrads()
    model.cleargrads()
    loss.backward()

    
    for p in model.params():
        p.data -= lr*p.grad.data

    if i % 100 == 0:
        print(loss)

# 将上面的步骤抽象为一种更便捷的方法：将模型定义为一个继承Layer类的类：
#%%
class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        #定义模型的网络结构
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y

# toy data: our old friend sin()
np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2*np.pi*x) + np.random.rand(100,1)

hidden_size = 5
out_size = 3
lr = 0.2
iters = 1000
model = TwoLayerNet(hidden_size,out_size)
model.plot(x)
# model.cleargrads()

for i in range(iters):
    #forward
    y_pred = model(x)
    # cal loss
    loss = F.mean_squared_error(y, y_pred)
    #cleargrads before backwards
    model.cleargrads()
    loss.backward()
    #更新模型参数
    for p in model.params():
        p.data -= lr*p.grad.data

    if i % 100 == 0:
        print(loss)

# 测试 MLP模型
import numpy as np
from dezero import Variable, Layer
from dezero.models import Model, MLP
import dezero.functions as F
import dezero.layers as L
model = MLP((10,20,30,40,1)) #5层（4个hidden1 个out）
# 在传入数据前模型不会初始化参数，参数全都是None
# next(model.params())

# 传入toy data
np.random.seed(0)
x = np.random.rand(100,1)
y_pred = model(x)
print(y_pred)
model._params #{'l0', 'l1', 'l2', 'l3', 'l4'}
model.__dict__['l0'].W.shape # (1, 10)


for i, w in enumerate(model.params()):
    print(f'{i}-{w.shape}: {w}')


# 测试__setattr__以及__dicit__的行为
#%%
class nametest:
    def __init__(self, name='name1'):
        self.name = name

tmp = nametest()
tmp.__dict__ # out: {'name': 'name1'}

# 从上面这个例子可以看出，实例tmp的所有变量的值都会被__setattr__以{name:value}的形式存到__dict__中

# 当set中有不同的数据结构时，会产生随机性:每次重启kernel都不一样
tmp2 = iter(set([1,3,2,4,'5','6']))
next(tmp2)
tmp = set([1,3,2,4,'5','6'])

def fun(tmp):
    for i in tmp:
        print(i)
        yield i

def func(tmp):
    for i in tmp:
        print(i)
        yield from fun(tmp)

tmp3 = func(tmp)
next(tmp3)


