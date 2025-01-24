

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
print(tokenizer.encode(text)) # error: hello not in the training set


