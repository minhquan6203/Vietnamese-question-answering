from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.t5_embedding import T5_Embedding
from encoder_module.init_encoder import build_uni_modal_encoder,build_multi_modal_encoder

class T5_Model(nn.Module):
    def __init__(self,config: Dict, num_labels: int):
        super(T5_Model, self).__init__()
        self.intermediate_dims = config["model"]["intermediate_dims"]
        #self.text_embbeding1 = Text_Embedding(config,256)
        #self.text_embbeding2 = Text_Embedding(config,32) 
        self.text_embbeding = T5_Embedding(config)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_labels = num_labels
        self.dropout=config["model"]["dropout"]
        self.classifier = nn.Linear(self.intermediate_dims, self.num_labels)
        self.attention_weights = nn.Linear(self.intermediate_dims, 1)
        self.multi_encoder = build_multi_modal_encoder(config)
        self.uni_encoder = build_uni_modal_encoder(config)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, id1_text: List[str], id2_text: List[str], labels: Optional[torch.LongTensor] = None):
        embbed, mask = self.text_embbeding(id1_text,id2_text)
        encoded_feature = self.uni_encoder(embbed, mask)
        feature_attended = self.attention_weights(torch.tanh(encoded_feature))
        
        attention_weights = torch.softmax(feature_attended, dim=1)
        feature_attended = torch.sum(attention_weights * encoded_feature, dim=1)
        
        logits = self.classifier(feature_attended)
        logits = F.log_softmax(logits, dim=-1)
        if labels is not None:
            loss = self.criterion(logits, labels)
            return logits,loss
        else:
            return logits

def createT5_Model(config: Dict, answer_space: List[str]) -> T5_Model:
    return T5_Model(config, num_labels=len(answer_space))