# coding:utf-8
import torch
import numpy as np
import json
import opennre
from opennre import encoder, model, framework
from extract_feature import BertVector
ckpt = 'ckpt/my_check_point.pth.tar'
rel2id = json.load(open('benchmark/exdata_new/rel2id.json',encoding='UTF-8'))

sentence_encoder = BertVector()

model = opennre.model.MyBagAttention(sentence_encoder, len(rel2id), rel2id)
framework = opennre.framework.MyBagRE(
    train_path='benchmark/exdata_new/train.txt',
    val_path='benchmark/exdata_new/val.txt',
    test_path='benchmark/exdata_new/val.txt',
    model=model,
    ckpt=ckpt,
    batch_size=160,
    max_epoch=60,
    lr=0.5,
    weight_decay=0,
    opt='sgd')
# Train
framework.train_model()
# Test
framework.load_state_dict(torch.load(ckpt)['state_dict'])
result = framework.eval_model(framework.test_loader)
print('AUC on test set: {}'.format(result['auc']))
