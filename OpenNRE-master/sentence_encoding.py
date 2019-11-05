from extract_feature import BertVector
import json

bert = BertVector()
sentence_list=["你好，你是我最喜欢的女孩子","你好呀"]
v = bert.encode(sentence_list)
print(v)