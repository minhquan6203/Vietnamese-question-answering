import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict, Optional
from data_utils.vocab import create_vocab

def Gpt2_tokenizer(config):
    tokenizer = GPT2Tokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
    special_tokens = {
    "bos_token": "<BOS>",     # Beginning of sentence token
    "eos_token": "<EOS>",     # End of sentence token
    "pad_token": "<PAD>",     # Padding token (if needed for batch processing)
    "unk_token": "<UNK>",     # Unknown token (if needed for out-of-vocabulary words)
    }
    tokenizer.add_special_tokens(special_tokens)
    if config["text_embedding"]["add_new_token"]:
        new_tokens,_ = create_vocab(config)
        new_tokens = set(new_tokens) - set(tokenizer.get_vocab().keys())
        tokenizer.add_tokens(list(new_tokens))
    return tokenizer

def Gpt2_Embedding(config):
    tokenizer = Gpt2_tokenizer(config)
    embedding = GPT2LMHeadModel.from_pretrained(config["text_embedding"]["text_encoder"])
    embedding.resize_token_embeddings(len(tokenizer))
    # freeze all parameters of pretrained model
    if config['text_embedding']['freeze']:
        for param in embedding.parameters():
            param.requires_grad = False
    return embedding

class Gpt2_Encode_Feature(nn.Module):
    def __init__(self, config):
        super(Gpt2_Encode_Feature, self).__init__()
        self.tokenizer=Gpt2_tokenizer(config)
        self.padding = config["tokenizer"]["padding"]
        self.max_input_length = config["tokenizer"]["max_input_length"]
        self.max_question_length = config['tokenizer']['max_question_length']
        self.truncation = config["tokenizer"]["truncation"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, context: List[str], question: List[str]=None):
        if question is None:
            encoded_inputs = self.tokenizer(
                                context,
                                padding= self.padding,
                                max_length=self.max_input_length,
                                truncation=self.truncation,
                                return_tensors='pt').to(self.device)
            encodings = {
                'input_ids': encoded_inputs.input_ids,
                'attention_mask': encoded_inputs.attention_mask,
                'labels': encoded_inputs.input_ids,
            }
        else:
            encoded_inputs = self.tokenizer(
                                question,
                                padding= self.padding,
                                max_length=self.max_question_length,
                                truncation=self.truncation,
                                return_tensors='pt').to(self.device)
            encodings = {
                'input_ids': encoded_inputs.input_ids,
                'attention_mask': encoded_inputs.attention_mask
            }
        return encodings