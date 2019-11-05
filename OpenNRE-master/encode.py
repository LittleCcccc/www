from extract_feature import BertVector

bert = BertVector()
list = ["犯罪以后自动投案，如实供述自己的罪行的，是自首。","犯罪以后自动投案，","犯罪以后自动投案，"]
v = bert.encode(list)
print(v)