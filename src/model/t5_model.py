from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.t5_embedding import T5_Embedding, T5_Encode_Feature, T5_tokenizer

class T5_Model(nn.Module):
    def __init__(self,config: Dict):
        super(T5_Model, self).__init__()
        self.embbeding = T5_Embedding(config)
        self.encode_feature = T5_Encode_Feature(config)
        self.tokenizer = T5_tokenizer(config)
        self.generator_args ={
            "max_length": config['generator_args']['max_length'],
            "num_beams": config['generator_args']['num_beams'],
            "length_penalty": config['generator_args']['length_penalty'],
            "no_repeat_ngram_size": config['generator_args']['no_repeat_ngram_size'],
            "early_stopping": config['generator_args']['early_stopping'],
        }

    def forward(self, input_text: List[str], answers: List[str]=None):
        inputs = self.encode_feature(input_text, answers)
        if answers is not None:
            outputs = self.embbeding(**inputs)
            return outputs.logits, outputs.loss
        else:
            pred_ids=self.embbeding.generate(**inputs,**self.generator_args)
            pred_tokens=self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            return pred_tokens