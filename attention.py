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

#===== part 2: adding trainable weights =====

# step 1: initialize
torch.manual_seed(123)
x_2 = inputs[1]
d_in=inputs.shape[1] # the input embeddin size = 3
d_out=2
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) #object type: torch.nn.parameter.Parameter
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# should all be vector of dim 2
query_2 = x_2@W_query
key_2 = x_2@W_key
value_2 = x_2@W_value

# step 2: compute attn scores:
# simply use query_2 to do dot product with all other input: 
# attn scores = q@k

# we first obtain all keys and values via matrix multiplication
keys = inputs @ W_key
values = inputs @ W_value

# attn scores of input 2
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

# we can generalize this computation to all keys to get the attn scores of input 2
attn_scores_2 = query_2@keys.T
# more general, attn scores to all inputs
inputs@W_query@keys.T # the 2nd row is the attn scores of input 2
print(attn_scores_2)

d_k = keys.shape[-1] # the embedding dim of keys
# attn_scores_2 is a vector, hence dim=-1 equal dim = 0
attn_weights_2 = torch.softmax(attn_scores_2/d_k**0.5, dim=-1) 
print(attn_weights_2)

# step 3: compute the context vector as a weighted sum over the value vectors
context_vec_2 = attn_weights_2@values
print(context_vec_2)


# final step: assemble
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self, x):
        keys = x@self.W_key
        queries = x@self.W_query
        values = x@self.W_value
        attn_scores = queries@keys.T #omega
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1) # here, attn_scores is a matrix, dim=-1 equal dim=1, meaning sum along row
        context_vec = attn_weights@values
        return context_vec
    
# test
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
# the 2nd row of result should equal to context_vec_2
print(sa_v1.forward(inputs)) # equivalent to sa_v1(inputs)

# v2: use nn.Linear instead of nn.Parameter to improve efficiency
#%%
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out,qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # nn.Linear is a class, with initialization with in_features and out_features, here are d_in and d_out, after initiating the nn.Linear, you can pass 
    def forward(self, x):
        keys = self.W_key(x) # same as self.W_key.forward(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries@keys.T #omega
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1) # here, attn_scores is a matrix, dim=-1 equal dim=1, meaning sum along row
        context_vec = attn_weights@values
        return context_vec
#%%
# test2
'''
v1 and v2 give different outputs because
they use different initial weights for the weight matrices since nn.Linear uses a more sophisticated weight initialization scheme.
'''
torch.manual_seed(123)
sa_v2 = SelfAttention_v2(d_in, d_out)
# the 2nd row of result should equal to context_vec_2
print(sa_v2.forward(inputs)) # different from that of v1x









# %%
