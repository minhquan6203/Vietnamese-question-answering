from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.gpt2_embedding import Gpt2_Embedding, Gpt2_Encode_Feature, Gpt2_tokenizer

class Gpt2_Model(nn.Module):
    def __init__(self,config: Dict):
        super(Gpt2_Model, self).__init__()
        self.embbeding = Gpt2_Embedding(config)
        self.encode_feature = Gpt2_Encode_Feature(config)
        self.tokenizer=Gpt2_tokenizer(config)
    def forward(self, question: List[str], context: List[str], start_idx, end_idx, answers: List[str]=None):
        inputs = self.encode_feature(question, context, start_idx, end_idx, answers)
        outputs = self.embbeding(**inputs)
        
        if answers is not None:
            return outputs.start_logits, outputs.end_logits, outputs.loss
        else:
            start_indices = torch.argmax(outputs.start_logits, dim=1)
            end_indices = torch.argmax(outputs.end_logits, dim=1) + 1
            
            pred_tokens_batch = []
            for i in range(len(question)):
                start_index = start_indices[i].item()
                end_index = end_indices[i].item()
                pred_tokens = self.tokenizer.decode(inputs["input_ids"][i][start_index:end_index],skip_special_tokens=True)
                pred_tokens_batch.append(pred_tokens)

            return pred_tokens_batch