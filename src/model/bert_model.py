from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.bert_embedding import Bert_Embedding, Bert_Encode_Feature,Bert_tokenizer

class Bert_Model(nn.Module):
    def __init__(self,config: Dict):
        super(Bert_Model, self).__init__()
        self.embbeding = Bert_Embedding(config)
        self.encode_feature = Bert_Encode_Feature(config)
        self.tokenizer=Bert_tokenizer(config)
    def forward(self, question : List[str], context: List[str], start_idx, end_idx ,answers: List[str]=None):
        inputs = self.encode_feature(question, context, start_idx, end_idx ,answers)
        outputs = self.embbeding(**inputs)
        if answers is not None:
            return outputs.logits, outputs.loss
        else:
            start_index=torch.argmax(outputs.start_logits)
            end_index = torch.argmax(outputs.end_logits) + 1
            pred_tokens = self.tokenizer.decode(inputs["input_ids"][0][start_index:end_index])
            return pred_tokens