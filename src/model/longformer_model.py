from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.longformer_embedding import Longformer_Embedding, Longformer_Encode_Feature,Longformer_tokenizer

class Longformer_Model(nn.Module):
    def __init__(self,config: Dict):
        super(Longformer_Model, self).__init__()
        self.embbeding = Longformer_Embedding(config)
        self.encode_feature = Longformer_Encode_Feature(config)
        self.tokenizer=Longformer_tokenizer(config)
    def forward(self, question: List[str], context: List[str], start_idx, end_idx, answers: List[str]=None):
        inputs = self.encode_feature(question, context, start_idx, end_idx, answers)
        outputs = self.embbeding(**inputs)
        
        if answers is not None:
            return outputs.start_logits, outputs.end_logits, outputs.loss
        else:
            start_indices = torch.argmax(outputs.start_logits, dim=1)
            end_indices = torch.argmax(outputs.end_logits, dim=1)
            
            pred_tokens_batch = []
            for i in range(len(question)):
                start_index = start_indices[i].item()
                end_index = end_indices[i].item()
                pred_tokens = self.tokenizer.decode(inputs["input_ids"][i][start_index:end_index],skip_special_tokens=True)
                pred_tokens_batch.append(pred_tokens)

            return pred_tokens_batch