import torch
import torch.nn as nn
from mask.masking import generate_padding_mask
class IDFVectorizer(nn.Module):
    def __init__(self, d_model, vocab, word_count):
        super(IDFVectorizer, self).__init__()
        self.vocab = vocab
        self.word_count = word_count
        self.idf_vector = self.compute_idf_vector()
        self.proj = nn.Linear(len(vocab), d_model)
    
    def compute_idf_vector(self):
        idf_vector = torch.zeros(len(self.vocab))
        for i, word in enumerate(self.vocab):
            if word in self.word_count:
                idf_value = torch.log(torch.tensor(len(self.word_count) / self.word_count[word]))
                idf_vector[i] = idf_value
        return idf_vector
    
    def compute_tf_vector(self, input_text):
        tf_vector = torch.zeros(len(self.vocab))
        total_words = len(input_text.split())
        
        for word in input_text.split():
            word=word.lower()
            if word in self.vocab:
                tf_vector[self.vocab.index(word)] +=1
            else:
                tf_vector[self.vocab.index("[unknown]")] +=1

        return tf_vector / total_words
    
    def forward(self, input_texts):
        tf_idf_vectors = []
        for input_text in input_texts:
            tf_vector = self.compute_tf_vector(input_text)
            tf_idf_vectors.append(tf_vector*self.idf_vector)
        tf_idf_vectors = torch.stack(tf_idf_vectors, dim=0)
        tf_idf_vectors = tf_idf_vectors.to(self.proj.weight.device)  # Chuyển đổi sang cùng device với self.proj
        features = self.proj(tf_idf_vectors).unsqueeze(1)
        padding_mask = generate_padding_mask(features, padding_idx=0)
        return features, padding_mask