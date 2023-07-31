from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.t5_embedding import T5_Embedding, Encode_Feature

class T5_Model(nn.Module):
    def __init__(self,config: Dict):
        super(T5_Model, self).__init__()
        self.embbeding = T5_Embedding(config)
        self.encode_feature = Encode_Feature(config)
    def forward(self, input_text: List[str], answers: List[str]=None):
        inputs = self.encode_feature(input_text, answers)
        if answers is not None:
            outputs = self.embbeding(**inputs)
            return outputs.logits, outputs.loss
        else:
            pred_ids=self.embbeding.generate(**inputs)
            return pred_ids

def createT5_Model(config: Dict) -> T5_Model:
    return T5_Model(config)