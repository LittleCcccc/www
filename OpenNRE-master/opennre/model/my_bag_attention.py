import torch
from torch import nn, optim
from .base_model import BagRE


class MyBagAttention(BagRE):
    """
    Instance attention for bag-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc = nn.Linear(768, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id =rel2id

    def infer(self, bag):
        """
        Args:
            bag: bag of sentences with the same entity pair
                [{
                  'text' or 'token': ...,
                  'h': {'pos': [start, end], ...},
                  't': {'pos': [start, end], ...}
                }]
        Return:
            (relation, score)
        """
        #TODO:
        return (rel, score)

    def forward(self, label, scope, token, pos1, pos2, train=True):
        """
        Args:
            label: (B), label of the bag
            scope: (B), scope for each bag
            token: (nsum), text
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(token)  # (nsum, H)

        # Attention
        if train:
            bag_rep = []
            query = torch.zeros((rep.size(0))).long()
            if torch.cuda.is_available():
                query = query.cuda()
            for i in range(len(scope)):
                query[scope[i][0]:scope[i][1]] = label[i]
            att_mat = self.fc.weight.data[query]  # (nsum, H)
            att_score = (rep * att_mat).sum(-1)  # (nsum)
            for i in range(len(scope)):
                bag_mat = rep[scope[i][0]:scope[i][1]]  # (n, H)
                softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]])  # (n)
                bag_rep.append(torch.matmul(softmax_att_score.unsqueeze(0), bag_mat).squeeze(
                    0))  # (1, n) * (n, H) -> (1, H) -> (H)
            bag_rep = torch.stack(bag_rep, 0)  # (B, H)
            # bag_rep = self.drop(bag_rep)
            bag_logits = self.fc(bag_rep)  # (B, N)
        else:#eval
            bag_logits = []
            att_score = torch.matmul(rep, self.fc.weight.data.transpose(0, 1))  # (nsum, H) * (H, N) -> (nsum, N)
            for i in range(len(scope)):
                bag_mat = rep[scope[i][0]:scope[i][1]]  # (n, H)
                softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]].transpose(0, 1))  # (N, (softmax)n)
                rep_for_each_rel = torch.matmul(softmax_att_score, bag_mat)  # (N, n) * (n, H) -> (N, H)
                logit_for_each_rel = self.softmax(self.fc(rep_for_each_rel))  # ((each rel)N, (logit)N)
                logit_for_each_rel = logit_for_each_rel.diag()  # (N)
                bag_logits.append(logit_for_each_rel)
            bag_logits = torch.stack(bag_logits, 0)  # after **softmax**

        return bag_logits
