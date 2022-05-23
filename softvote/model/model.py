"""
"""
import torch
from torch import nn
import transformers
from sklearn.metrics import f1_score
from transformers import ElectraModel, ElectraTokenizer
from transformers import FunnelTokenizerFast, FunnelModel
from transformers import BertTokenizerFast, BertModel


class ElectraSummarizer(nn.Module):

    def __init__(self):
        """
        """
        super(ElectraSummarizer, self).__init__()
        self.encoder = ElectraModel.from_pretrained("monologg/koelectra-small-v3-discriminator")
        self.fc = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, segs, clss, mask, mask_clss):
        """
        """
        top_vec = self.encoder(input_ids = x.long(), attention_mask = mask.float(),  token_type_ids = segs.long()).last_hidden_state
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss.long()]
        sents_vec = sents_vec * mask_clss[:, :, None].float()
        h = self.fc(sents_vec).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_clss.float()
        return sent_scores


class FunnelSummarizer(nn.Module):

    def __init__(self):
        """
        """
        super(FunnelSummarizer, self).__init__()
        self.encoder = FunnelModel.from_pretrained("kykim/funnel-kor-base")
        self.fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, segs, clss, mask, mask_clss):
        """
        """
        top_vec = self.encoder(input_ids = x.long(), attention_mask = mask.float(),  token_type_ids = segs.long()).last_hidden_state
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss.long()]
        sents_vec = sents_vec * mask_clss[:, :, None].float()
        h = self.fc(sents_vec).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_clss.float()
        return sent_scores


class BertSummarizer(nn.Module):

    def __init__(self):
        """
        """
        super(BertSummarizer, self).__init__()
        self.encoder = BertModel.from_pretrained("kykim/bert-kor-base")
        self.fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, segs, clss, mask, mask_clss):
        """
        """
        top_vec = self.encoder(input_ids = x.long(), attention_mask = mask.float(),  token_type_ids = segs.long()).last_hidden_state
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss.long()]
        sents_vec = sents_vec * mask_clss[:, :, None].float()
        h = self.fc(sents_vec).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_clss.float()
        return sent_scores
