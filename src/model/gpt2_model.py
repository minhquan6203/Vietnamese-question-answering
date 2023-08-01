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
        self.tokenizer = Gpt2_tokenizer(config)
        self.generator_args ={
            'max_length': config['generator_args']['max_length'],
            'min_length': config['generator_args']['min_length'],
            'num_beams': config['generator_args']['num_beams'],
            'length_penalty': config['generator_args']['length_penalty'],
            'no_repeat_ngram_size': config['generator_args']['no_repeat_ngram_size'],
            'early_stopping': config['generator_args']['early_stopping'],
            'num_return_sequences': config['generator_args']['num_return_sequences'],
            'do_sample':config['generator_args']['do_sample'],
            'top_k':config['generator_args']['top_k']
        }

    def forward(self, context: List[str], question: List[str]=None):
        inputs = self.encode_feature(context, question)
        if question is None:
            outputs = self.embbeding(**inputs)
            return outputs.logits, outputs.loss
        else:
            pred_ids=self.embbeding.generate(**inputs,**self.generator_args)
            pred_tokens=self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            return pred_tokens