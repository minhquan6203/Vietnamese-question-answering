import torch
from torch import nn
from torch.nn import functional as F
from transformers import RobertaTokenizer,RobertaForQuestionAnswering
from typing import List, Dict, Optional
from data_utils.vocab import create_vocab

def Roberta_tokenizer(config):
    tokenizer = RobertaTokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
    if config["text_embedding"]["add_new_token"]:
        new_tokens,_ = create_vocab(config)
        new_tokens = set(new_tokens) - set(tokenizer.get_vocab().keys())
        tokenizer.add_tokens(list(new_tokens))
    return tokenizer

def Roberta_Embedding(config):
    if config["text_embedding"]["add_new_token"]:
        tokenizer = Roberta_tokenizer(config)
        embedding = RobertaForQuestionAnswering.from_pretrained(config["text_embedding"]["text_encoder"])
        embedding.resize_token_embeddings(len(tokenizer))
    else:
        embedding = RobertaForQuestionAnswering.from_pretrained(config["text_embedding"]["text_encoder"])
    # freeze all parameters of pretrained model
    if config['text_embedding']['freeze']:
        for param in embedding.parameters():
            param.requires_grad = False
    return embedding

class Roberta_Encode_Feature(nn.Module):
    def __init__(self, config):
        super(Roberta_Encode_Feature, self).__init__()
        self.tokenizer = Roberta_tokenizer(config)
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
            start_idx = torch.tensor([context[i].find(answers[i]) for i in range(len(answers))])
            end_idx = torch.tensor([context[i].find(answers[i]) + len(answers[i]) -1 for i in range(len(answers))])
            start_idx.clamp_(0, encoded_inputs.input_ids.size(1) - 1)
            end_idx.clamp_(0, encoded_inputs.input_ids.size(1) - 1)
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