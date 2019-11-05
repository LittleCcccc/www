import json
import re
file = open("./data/2000_result.json",'r',encoding='utf-8')
dict = json.load(file)


id_dict = {}
for item in dict['term']:
    #print(item)
    cnt = 0
    content = item['content'][0]['content']
    list = re.split("[。，\n]",content)
    for i in list:
        if len(i)!=0:
            id_dict[cnt]={}
            id_dict[cnt]['sentence']=i
            cnt += 1

print(id_dict)
file = open("./data/iddict2.json",'w',encoding='utf-8')
json.dump(id_dict,file,ensure_ascii=False)