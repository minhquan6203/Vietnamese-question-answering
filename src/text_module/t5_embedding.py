import torch
from torch import nn
from torch.nn import functional as F
from transformers import T5Tokenizer,T5ForConditionalGeneration
from typing import List, Dict, Optional
from data_utils.vocab import create_vocab

def T5_tokenizer(config):
    tokenizer = T5Tokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
    if config["text_embedding"]["add_new_token"]:
        new_tokens,_ = create_vocab(config)
        new_tokens = set(new_tokens) - set(tokenizer.get_vocab().keys())
        tokenizer.add_tokens(list(new_tokens))
    return tokenizer

def T5_Embedding(config):
    if config["text_embedding"]["add_new_token"]:
        tokenizer = T5_tokenizer(config)
        embedding = T5ForConditionalGeneration.from_pretrained(config["text_embedding"]["embeding_name"])
        embedding.resize_token_embeddings(len(tokenizer))
    else:
        embedding = T5ForConditionalGeneration.from_pretrained(config["text_embedding"]["embeding_name"])
        # freeze all parameters of pretrained model
    if config['text_embedding']['freeze']:
        for param in embedding.parameters():
            param.requires_grad = False
    return embedding

class T5_Encode_Feature(nn.Module):
    def __init__(self, config):
        super(T5_Encode_Feature, self).__init__()
        self.tokenizer=T5_tokenizer(config)
        self.padding = config["tokenizer"]["padding"]
        self.max_input_length = config["tokenizer"]["max_input_length"]
        self.max_target_length = config["tokenizer"]["max_target_length"]
        self.truncation = config["tokenizer"]["truncation"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input_text: List[str], answers: List[str]=None):
        encoded_inputs = self.tokenizer(
                                input_text,
                                padding= self.padding,
                                max_length=self.max_input_length,
                                truncation=self.truncation,
                                return_tensors='pt',
                            ).to(self.device)
        if answers is not None:
            encoded_targets = self.tokenizer(
                                    answers,
                                    padding= self.padding,
                                    max_length=self.max_target_length,
                                    truncation=self.truncation,
                                    return_tensors='pt',
                                ).to(self.device)
            encoded_targets[encoded_targets == self.tokenizer.pad_token_id] = -100
            encodings = {
                'input_ids': encoded_inputs.input_ids,
                'attention_mask': encoded_inputs.attention_mask,
                'labels': encoded_targets.input_ids,
                'decoder_attention_mask': encoded_targets.attention_mask,
            }
        else:
            encodings = {
                'input_ids': encoded_inputs.input_ids,
                'attention_mask': encoded_inputs.attention_mask
            }
        return encodings