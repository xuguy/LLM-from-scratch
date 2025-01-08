import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
    )

# for 1 query

# step 1: calculate attension scores:
# pick the 2nd input as "query"
query = inputs[1]
# initialize a vector(tensor) to store the result
# torch.empty() is similar to np.empty()
# inputs.shape: torch.Size([6, 3])
attn_scores_2 = torch.empty(inputs.shape[0])

#把inputs的每一行x_i拿出来和query做点乘
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i,query)
print(attn_scores_2)

# above for loop equivalent to:
attn_scores_2 = inputs@query

# step 2: normalize (sum to 1, make it probabalistic)

# example: simple sum
attn_weights_2_tmp = attn_scores_2/attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# softmax naive way (take exp element-wise then sum)
def softmax_naive(x):
    return torch.exp(x)/torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# softmax torch way
# 注意，softmax里面dim=0，因为attn_scores_2是一个向量，向量默认为矩阵的列向量，要对矩阵的列向量做softmax就是dim=0 （类似axis=0）
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# step 3: get context vector
query = inputs[1]

# initialize an empty vec to store results
context_vec_2 = torch.zeros(query.shape)

# weighted (attn weight) sum of all input tokens
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i


# for all query
attn_scores = torch.empty(6,6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i,j] = torch.dot(x_i,x_j)
print(attn_scores)

# above equivalent to:
# attn_scores_2 is the 2nd row of attn_scores
attn_scores = inputs@inputs.T

# softmax (maxtrix form)

# 注意，这里dim=-1等价于2维矩阵的dim=1，为什么之前softmax的dim=0而这里是dim=1？因为之前仅仅计算一个query，attn_scores_2是一个列向量，但是在这里计算所有querys，计算结果attn_scores是一个矩阵，原来的列向量是这个矩阵的行向量，即attn_scores_2是attn_scores的第二行
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

# verify:
row_2_sum = torch.sum(attn_weights[1])
row_sum = torch.sum(attn_weights, dim=1)
print(row_2_sum)
print(row_sum)

# get context matrix(calculate and stack all context vec)
all_context_vec = attn_weights@inputs



