import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertTokenizer,BertForQuestionAnswering
from typing import List, Dict, Optional
from data_utils.vocab import create_vocab

def Bert_tokenizer(config):
    tokenizer = BertTokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
    if config["text_embedding"]["add_new_token"]:
        new_tokens,_ = create_vocab(config)
        new_tokens = set(new_tokens) - set(tokenizer.get_vocab().keys())
        tokenizer.add_tokens(list(new_tokens))
    return tokenizer

def Bert_Embedding(config):
    if config["text_embedding"]["add_new_token"]:
        tokenizer = Bert_tokenizer(config)
        embedding = BertForQuestionAnswering.from_pretrained(config["text_embedding"]["text_encoder"])
        embedding.resize_token_embeddings(len(tokenizer))
    else:
        embedding = BertForQuestionAnswering.from_pretrained(config["text_embedding"]["text_encoder"])
    # freeze all parameters of pretrained model
    if config['text_embedding']['freeze']:
        for param in embedding.parameters():
            param.requires_grad = False
    return embedding

class Bert_Encode_Feature(nn.Module):
    def __init__(self, config):
        super(Bert_Encode_Feature, self).__init__()
        self.tokenizer = Bert_tokenizer(config)
        self.padding = config["tokenizer"]["padding"]
        self.max_input_length = config["tokenizer"]["max_input_length"]
        self.truncation = config["tokenizer"]["truncation"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, question : List[str], context: List[str], start_idx: torch.LongTensor, end_idx: torch.LongTensor ,answers: List[str]=None):
        encoded_inputs = self.tokenizer(
                                question, context,
                                padding = self.padding,
                                max_length = self.max_input_length,
                                truncation = self.truncation,
                                return_tensors='pt',
                            ).to(self.device)
        if answers is not None:
            context_lengths = [len(c) for c in context]
            for i in range(len(end_idx)):
                if end_idx[i] >= context_lengths[i]:
                    end_idx[i] = context_lengths[i] - 1
                if start_idx[i] >= context_lengths[i]:
                    start_idx[i] = 0

            encodings = {
                'input_ids': encoded_inputs.input_ids,
                'attention_mask': encoded_inputs.attention_mask,
                'start_positions': start_idx.to(self.device),
                'end_positions': end_idx.to(self.device),
            }
        else:
            encodings = {
                'input_ids': encoded_inputs.input_ids,
                'attention_mask': encoded_inputs.attention_mask
            }
        return encodings