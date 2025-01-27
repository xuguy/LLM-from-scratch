

with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
print(f'total number of character: {len(raw_text)}')
print(raw_text[:99]) # pring the first n characters

import re
text = 'Hello, world. This, is a text'
result = re.split(r'(\s)', text) # split by \s: space
print(result)

result = re.split(r'([,.]|\s)', text)
print(result)

result = [item for item in result if item.strip()]
print(result)

text = "Hello, world. Is this-- a test?"
# this can be a simple tokenizer for the-verdict
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed)) # without whitespace
print(preprocessed[:30])

# map all words (a vocabulary: preprocessced) to integer
# sorted: alphabetically
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token: integer for integer, token in enumerate(all_words)}

# test_dict = {'1':1, '2':2}
# test_dict.items()

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

# a tokenizer (complete) class
class SimpleTokenizerV1:

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = ' '.join([self.int_to_str[i] for i in ids])

        # remove spaces before the specified punctuation
        # sub for substitude
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# test with our tokenizer
tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""

ids = tokenizer.encode(text)

# test decode
print(tokenizer.decode(ids))

# test with text not included in the training set
text = 'Hello, do you like tea?'
print(tokenizer.encode(text)) # error: hello not in the training set, need to modify our class to accomodate

# adding  <|unk|> <|endoftext|>
# why wrap a list to set()? because only list can be extended
# 'set' object has no attribute 'append'
# remember: preprocessed is a list of words(tokens)
all_tokens = sorted(list(set(preprocessed)))
# .append() is used to add a single element to the end of a list
# while .extend() can add multiple at once
all_tokens.extend(['<|endoftext|>', '<|unk|>'])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
print(len(vocab.items())) # should be 1132, previous 1130
# print the last 5 items
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)


# following is the new class:
class SimpleTOkenizerV2:
    def __init__(self, vocab):
        # create a 正反互查表
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    # encode: 把string转化为integer（ids）
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else '<|unk|>' for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]

        return ids
    
    def decode(self, ids):
        # 把一串ids连接起来，用空格分开
        text = ' '.join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)

        return text

text1 = 'Hello, do you like tea?'
text2 = 'In the sunlit terraces of the palace.'
text = ' <|endoftext|> '.join((text1, text2))
print(text)

tokenizer = SimpleTOkenizerV2(vocab)
print(tokenizer.encode(text))

print(tokenizer.decode(tokenizer.encode(text)))

# byte pair encoding
from importlib.metadata import version
import tiktoken
print('tiktoken version:', version('tiktoken')) #0.8.0

tokenizer = tiktoken.get_encoding('gpt2')

text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces")
integers = tokenizer.encode(text, allowed_special={'<|endoftext|>'}) 
print(integers)

strings = tokenizer.decode(integers)
print(strings)


# load data
with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f'x: {x}')
print(f'y:      {y}')

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, '---->', desired)

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), '---->', tokenizer.decode([desired]))

# efficient data loader
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        # tokenize the entire text
        token_ids = tokenizer.encode(txt)

        # use a slide window to chunk the book into overlapping sequences of max_length
        # max_length: context_size
        # stride: how many steps the window should move in a time
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i+1: i + max_length +1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    # return the total number of rows in the dataset
    def __len__(self):
        return len(self.input_ids)
    
    # return a single row from the dataset
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size = 4, max_length=256, stride=128, shuffle=True, drop_last = True, num_workers=0):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last, num_workers = num_workers)

    return dataloader

with open('the-verdict.txt', 'r', encoding = 'utf-8') as f:
    raw_text = f.read()

# stride =1 means the window slide 1 step at a time
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

# batchsize >1
# note that max_length = stride =4, this ensure we use the dataset fully and avoids any overlap between the batches since more overlap could lead to increased overfitting.
dataloader = create_dataloader_v1(raw_text, batch_size = 8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print('Inputs:\n', inputs)
print('\nTargets:\n', targets)

inputs_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3])))
print(embedding_layer(inputs_ids))

# a more realistic input size words embedding
vocab_size = 50257
output_dim = 256
# input_vecter@embedding_weight_matrix
# torch.nn.Embedding will initialize a matrix (weights, initialized with random values) dedicated for words embedding.
# these torch.nn.Embedding module are designed for later optimization with training.
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# load data and tokenize data
max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride = max_length, shuffle = False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print('Token IDs:\n', inputs)
print('\nInpyts shape:\n', inputs.shape)

# 注意，token_embedding_layer输入inputs以后，inputs原来存的是ids，token_embedding_layer会先把inputs里面存的ids转换成one-hot以后再和weights matrix做矩阵运算，因此我们之前在实例化token_embedding_layer = torch.nn.Emebedding要输入vocabsize，也就是50257，这样，如果inputs里面储存的ids=52，那么他就会先被转换成一个50257维的onehot-vector（其中第52个位置为1，其他为0）后再进行矩阵运算。
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# create absolute position embedding

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
# remember that context_length = 4, 我们对每一个输入向量都做position embedding，position的长度和context_length一致
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

# 直接把position embedding matrix加到embedding matrix上完成位置嵌入
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)