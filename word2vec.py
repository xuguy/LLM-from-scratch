text = 'You say goodbye and I say hello.'
text = text.lower()
text = text.replace('.',' .')

words = text.split(' ')



word_to_id = {}
id_to_word = {}

for word in words:
    if word not in word_to_id:
        new_id = len(word_to_id)
        word_to_id[word] = new_id
        id_to_word[new_id] = word


# 准备语料库
import numpy as np
corpus = [word_to_id[w] for w in words]
corpus = np.array(corpus)

# pack the above code in one function
def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

# test:
text = 'Your say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)    

# distributed hypothesis
import sys
import numpy as np

sys.path.append('DL-code-nlp')
from common.util import preprocess

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)


# 共现矩阵
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype = np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix

# measure similarity

def cos_similarity(x, y, eps = 1e-8):
    nx = x/np.sqrt(np.sum(x**2) + eps)
    ny = y/np.sqrt(np.sum(y**2) + eps)
    return np.dot(nx, ny)

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id['you']]
# 注意，preprocess会把所有单词转换为小写
c1 = C[word_to_id['i']]
print(cos_similarity(c0, c1))

# 相似度排序
def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print(f'{query} is no found')
        return
    
    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # cal cos similarity
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue

        print(f'{id_to_word[i]}: {similarity[i]}')

        count += 1
        if count >= top:
            return

# test:
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

most_similar('you', word_to_id, id_to_word, C, top=5)

# word2vec

import numpy as np
import sys
sys.path.append('DL-code-nlp')
# MatMul is a layer, which include forward and backward method
from common.layers import MatMul
c = np.array([[1,9,9,9,9,9,9]])
W = np.random.rand(7, 3)
layer = MatMul(W)
h = layer.forward(c)

print(h)

# cbow
# forward
c0 = np.array([[1,0,0,0,0,0,0]])
c1 = np.array([[0,0,1,0,0,0,0]])

W_in = np.random.rand(7,3)
W_out = np.random.rand(3, 7)

in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)

h = 0.5*(h0+h1)
s = out_layer.forward(h)

print(s)


# stable softmax:
# to avoid the overflow of exponential fucntion
'''
check:
https://www.parasdahal.com/softmax-crossentropy
'''
def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


# create contexts and target
def create_contexts_target(corpus, window_size=1):

    # 指定target的范围，即第二个至倒数第二个
    target = corpus[window_size:-window_size]
    contexts = []

    # 下面的这个range指的是窗口内的词的可能取值；例如，这里window_size=1,那么range就是从range(1, 6):1,2,3,4,5，因为len(corpus)=7,因此corpus的最大index为6，那么target的range最多到5，否则contexts的shape就不合规则
    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            # t==0即为target，跳过
            if t == 0:
                continue
            cs.append(corpus[idx+t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)

contexts, target = create_contexts_target(corpus, window_size=1)
print(f'contexts: \n{contexts}\ntarget: \n{target}')

# switch into one-hot representation
from common.util import convert_one_hot
from common.layers import MatMul, SoftmaxWithLoss

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

contexts, target = create_contexts_target(corpus, window_size=1)

vocab_size= len(word_to_id)
contexts = convert_one_hot(contexts, vocab_size)
target = convert_one_hot(target, vocab_size)


class SimpleCBOW: # contexts size = 1
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        # 使用32位浮点数初始化
        W_in = 0.01*np.random.randn(V,H).astype('f')
        W_out = 0.01*np.random.rand(H,V).astype('f')
        # 输入侧上下文的MatMul层的数量与上下文的单词数相同
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()
        # 序列化layers，让输入数据按顺序通过各层
        layers = [self.in_layer0, self.in_layer1, self.out_layer]

        #保存计算的权重（参数）和梯度
        self.params, self.grads = [], []

        for layer in layers:
            # 其实就是append
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])

        h=0.5*(h0+h1)
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss
    
    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5 # 有2个不同的输入context向量,forward结果相加并*0.5，因此backward需要把结果x0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None

# test
from common.trainer import Trainer
from common.optimizer import Adam
window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000
model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)

trainer.plot()

# get word embedding
word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])



