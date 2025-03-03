from dezero.core import Parameter
import weakref, os
import numpy as np
import dezero.functions as F
from dezero import cuda


# base class
class Layer:
    # Layer这个基类定义了一些所有Layer都会有的attribute和method，例如为了方便管理参数而设定的params()方法，cleargrads()方法
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        # __setattr__是一个在实例化Layer时自动调用的方法，我们这里重新定义了__setattr__的逻辑，当实例化Layer后，该实例也会自动调用这个方法，虽然我们重写了这个方法的逻辑，但后面又继承了base类的__setattr__方法，因此__setattr__方法依旧会起效，只是我们规定，在他起效前增加一个判断语句
        # 另外，一个class的所有实例变量都会被__setattr__自动以字典形式存到实例变量__dict__中，其中实例变量的名字为name，变量的值为value，以{name:value}形式保存
        # 只有当value是Parameter实例时才向self._params增加name
        if isinstance(value, (Parameter, Layer)):
            # Parameter和Layer类自身都可以作为参数
            self._params.add(name)
        #只有Parameter类实例变量的name会被添加到self._params
        #但所有实例变量都会被添加到__dict__中，到时候按需要取出即可
        super().__setattr__(name, value)

    def __call__(self, *inputs):

        # forward方法将会在继承Layer类的子类中实现
        # 这里可以与Function类的实现做一个对比，Function类中，如果不需要反向传播（例如推理模式），实例不会保留inputs和outputs，仅仅做计算并return outputs，然后所有局部变量（非self.var)就会被删除
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        #  弱引用：不增加被引用对象的引用计数，因此被引用对象（局部变量inputs/outputs）在用完后就会被删除(回收)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        # 
        return outputs if len(outputs)>1 else outputs[0]
    
    def forward(self, inputs):
        raise NotImplementedError()
    
    def params(self):
        # 按顺序逐个取出Layer实例所有的Parameter实例
        # 取出Layer实例_params中所有的参数，_params中的参数原来只有Parameter实例，现在又扩充了Layer实例
        for name in self._params:
            
            #取出参数obj
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                #如果参数obj是Layer实例，那么就从Layer实例obj中递归地取出所有参数
                # 我们通过 yield from 来使用一个生成器创建另一个新的生成器
                # print(name)
                yield from obj.params()
            else:
                #否则，也就是obj是Parameter实例，那么就从obj = self.__dict__[name]中取出参数
                yield obj

    def cleargrads(self):
        # reset all Parameters' grad
        for param in self.params():
            param.cleargrad()

    # cupy adaptation
    # 作用对象是Layer中的params()
    def to_cpu(self):
        for param in self.params():
            # param是继承自Variable类的Parameter类，把层中的param拿出来修改后，会直接改动层中的对应的param；且Variable类同Parameter类同样具备to_cpu方法
            param.to_cpu()
    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def _flatten_params(self, params_dict, parent_key=''):
        for name in self._params:

            obj = self.__dict__[name]

            # 命名规则：如果有子layer，那么用'/'分割parent和child，如果没有自layer（直接获取到的是Parameter），那么就直接用该parameter的名字
            key = parent_key + '/' + name if parent_key else name
            #如果取出的是Layer，则递归调用
            # 如果取出的是Parameter，那么就把该Parameter加到param_dict中
            if isinstance(obj, Layer):
                # 递归遍历
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):

        # make sure Parameters are all in cpu RAM, so that we could use np.savez
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)

        # 取出数据（np.ndarray），这样才能调用savez保存
        # 注意，param.data里面是一个np.array数组
        array_dict = {key: param.data for key, param in params_dict.items() if param is not None}

        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            # 检查path是否存在，如存在则删除
            if os.path.exists(path):
                os.remove(path)
            raise
    
    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        # _flatten_params是为了获取原Layer的结构，这要求我们要有和参数对应的一模一样的model
        # 用load来的数据逐个替换对应key名下的数据
        for key, param in params_dict.items():
            param.data = npz[key]

#作为Layer的Linear类，而不是作为函数的Linear类
class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__() #激活Layer的init
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name = 'W')
        # 上面先设定为None（self.W.data=None），在forward方法中再创建权重（延迟创建权重W的时间），这样就能自动确定Linear类的输入大小（in_size)而无需用户指定
        # 如果in_size是None，那么下面这个if判断判断为假，不处理_init_W()，也就是不初始化权重，留到forward里再初始化
        # 然后再给Parameter的name属性(attribute)标记上'W'
        # Parameter 的name属性继承自Variable，这样self.W这个Parameter类的实例就有一个name属性，我们后续就可以通过self.Parameter.__dict__筛选不同名字的Parameter实例
        if self.in_size is not None:
            self._init_W()
        
        if nobias:
            self.b = None

        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name = 'b')
            # print(self.b)
            # self.b.data
    
    # 初始化W的方法（初始化，即往原来是W.data==None的self.W中传入具体的非None数据）
    def _init_W(self, xp=np):
        # out_size已知，而in_size可能未知
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype)*np.sqrt(1/I)
        self.W.data = W_data

    def forward(self, x):
        # forward将会根据输入x的shape创建权重数据
        # x是被传入的数据，x.shape[1]就是Linear层参数的in_size
        # 我们只需要按照layer=Linear(100)的方式指定输出大小即可
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.linear(x, self.W, self.b)
        return y


    