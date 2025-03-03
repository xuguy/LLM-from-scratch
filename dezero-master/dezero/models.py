from dezero import Layer, utils
import dezero.functions as F
import dezero.layers as L
import numpy as np


# base class for Model
class Model(Layer):

    # 给models加入plot方法，直接画出计算图
    def plot(self, *inputs, to_file = 'model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose = True, to_file=to_file)
    

# MLP model: MLP又是全连接层(fc)神经网络的别名

class MLP(Model):
    def __init__(self, fc_output_sizes, activation = F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            # 把层设置为模型的实例变量来对层的参数进行管理
            # 给自己添加以 l1/l2/l3...命名的L.Linear的实例layer作为自己的属性（属性就是类似self.var这样的类变量，类变量会被保存到MLP.__dicit__中作为MLP的参数
            setattr(self, 'l'+str(i), layer)
            self.layers.append(layer)
    
    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        #最后一层的输出就是前向传播的结果
        return self.layers[-1](x)