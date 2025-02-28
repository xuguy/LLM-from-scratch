import math
import random
import numpy as np

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle = True):

        # dataset mush be a well-defined sub-class of Dataset class
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)

        # 向上取整，保证可以遍历完整个数据集
        self.max_iter = math.ceil(self.data_size / batch_size)

        self.reset()

    # 重新打乱顺序（重置iter），意思是本epoch结束
    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))

        else:
            self.index = np.arange(len(self.dataset))

    # 返回iterator对象，也就是class 本身
    def __iter__(self):
        return self
    
    # 定义返回下一个元素的方法
    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration
        
        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i*batch_size : (i+1)*batch_size]
        batch = [self.dataset[i] for i in batch_index]
        x = np.array([example[0] for example in batch])
        t = np.array([example[1] for example in batch])

        self.iteration += 1
        return x, t
    
    def next(self):
        return self.__next__()