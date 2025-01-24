

with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
print(f'total number of character: {len(raw_text)}')
print(raw_text[:10]) # pring the first n characters