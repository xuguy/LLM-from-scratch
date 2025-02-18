# wine score data example
import torch
import numpy as np
import csv
wine_path = '../dlwpt-code/data/p1ch4/tabular-wine/winequality-white.csv'

# skiprows=1: the first row should not be read since it contains the column names
wineq_numpy = np.loadtxt(wine_path, dtype = np.float32, delimiter = ';', skiprows = 1)
wineq_numpy

# check
# with next(), we read only the first row, which is col names
col_list = next(csv.reader(open(wine_path), delimiter = ';'))
wineq_numpy.shape, col_list

# proceed to convert numpy to torch.tensor:
wineq = torch.from_numpy(wineq_numpy)
wineq.shape, wineq.dtype

# wine data: containing the 11 variables associated with the chemical analysis
data = wineq[:, :-1]
data, data.shape

target = wineq[:, -1]
target, target.shape

# convert score to label: just int it
target = wineq[:, -1].long() # tensor([6, 6, 6,  ..., 6, 7, 6])

# or use one-hot encoding:
# create an empty tensor to store one hot representation
target_onehot = torch.zeros(target.shape[0], 10)
# 这段是如何做到onehot的呢？
# 第一个参数1，表示在dim=1进行操作
# 第二个参数，target.unsqueeze(1)把target转换成(n,1)的列向量
# 第三个参数：把target.unsqueeze(1)的dim=1（也就是每一行的元素）表示为一个第k个元素值为1.0的向量
'''
Tensor.scatter_(dim, index, src, *, reduce=None) → Tensor:
dim=1, index=target.unsqueeze(1), which is a batch of index, src=1.0: a scalar tensor,
Writes all values from the tensor src into self(which is target_onehot) at the indices specified in the index tensor. For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.
'''
target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

data_mean = torch.mean(data, dim = 0)
data_var = torch.var(data, dim = 0)

data_normalized = (data-data_mean) / torch.sqrt(data_var)
data_normalized

# 4.3.6 Finding threshold:
# goal: look at the data with an eye to seeing if there is an easy way to tell good and bad wines apart
# 筛选出一些行，这些行含有分数小于等于3的数据
# example of advanced indexing (you can ignore if you are already familiar with numpy, here is the same for tensor)
bad_indexes = target <= 3 # torch.Size([4898])
bad_indexes.shape, bad_indexes.dtype, bad_indexes.sum() # 有20瓶酒bad

bad_data = data[bad_indexes]
bad_data.shape

bad_data = data[target <= 3]
mid_data = data[(target > 3) & (target < 7)]
good_data = data[target >=7]

bad_mean = torch.mean(bad_data, dim = 0)
mid_mean = torch.mean(mid_data, dim = 0)
good_mean = torch.mean(good_data, dim = 0)

# 打印出好、中、差酒各个features 的平均值
for i, args in enumerate(zip(col_list, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))

'''
 0 fixed acidity          7.60   6.89   6.73
 1 volatile acidity       0.33   0.28   0.27
 2 citric acid            0.34   0.34   0.33
 3 residual sugar         6.39   6.71   5.26
 4 chlorides              0.05   0.05   0.04
 5 free sulfur dioxide   53.33  35.42  34.55
 6 total sulfur dioxide 170.60 141.83 125.25
 7 density                0.99   0.99   0.99
 8 pH                     3.19   3.18   3.22
 9 sulphates              0.47   0.49   0.50
10 alcohol               10.34  10.26  11.42
'''

# 利用total_sulfur_dioxide这个指标来区分好酒与坏/中酒（粗糙的指标）
total_sulfur_threshold = 141.83
total_sulfur_data = data[:,6]
predicted_indexes = torch.lt(total_sulfur_data, total_sulfur_threshold)
predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum()
# out: (torch.Size([4898]), torch.bool, tensor(2727))
# 意思是，只有2727个样本被预测为好，总样本数为4898

#接下来查看真实情况：
actual_indexes = target > 5
actual_indexes.shape, actual_indexes.dtype, actual_indexes.sum()
# out: (torch.Size([4898]), torch.bool, tensor(3258))
# 数量相当接近了，接下来进一步看看有多少预测和实际匹配上了：

# actual_indexes 必须和 predicted_indexes 同为True( true positive/ total prediction)
n_matches = torch.sum(actual_indexes & predicted_indexes).item()
n_predicted = torch.sum(predicted_indexes).item()
n_actual = torch.sum(actual_indexes).item()
n_matches, n_matches / n_predicted, n_matches / n_actual

# next start 4.5: representing text


