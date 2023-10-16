from sentence_transformers import util
import torch
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from underthesea import sent_tokenize
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import string
import re
from typing import List
import unicodedata
from pyvi.ViTokenizer import tokenize

# model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
# model = SentenceTransformer('intfloat/multilingual-e5-large')
# model = SentenceTransformer('keepitreal/vietnamese-sbert')

def drop_last_dot(sentence):
    if sentence[-1] == '.':
        sentence = sentence[:-1]
    return sentence

def preprocess_text(text: str) -> str:    
    text = re.sub(r"['\",\.\?:\-!]", "", text)
    text = text.strip()
    text = " ".join(text.split())
    text = text.lower()
    return text

# def split_sentence(paragraph):
#     paragraph=paragraph.replace(':\n\n',': ').strip()
#     pre_context_list = paragraph.split("\n\n")
#     context_list = []
#     for context in pre_context_list:
#         sen = context.split(". ")
#         context_list = context_list + [s for s in sen if len(s.split())>1]
#     context_list=[preprocess_text(c) for c in context_list]
#     return context_list

def split_sentence(paragraph):
    context_list=[]
    if paragraph[-2:] == '\n\n':
      context_list.append('\n\n')
      paragraph = paragraph[:-2]
    c = True
    start = 0
    while c:
      context = ""
      for i in range(start ,len(paragraph[:-2])):
        if paragraph[i] == "." and i != (len(paragraph) - 1):
          # Kiểm tra trường hợp "\n\n"
          if paragraph[i+1:i+2] == "\n":
            context = context + paragraph[i]
            start = i + 1
            break

          # Kiểm tra trường hợp gặp " "
          if paragraph[i+1] == " ":
            # Nếu sau " " không phải chữ thường thì ngưng
            if not paragraph[i+2].islower():
              context = context + paragraph[i]
              start = i + 2
              break
        context = context + paragraph[i]
        if i == len(paragraph[:-3]):
          start = i
      if start == len(paragraph[:-3]):
        context += paragraph[start+1:]
        c = False
      context = preprocess_text(context)
      if len(context.split()) > 1:
        context_list.append(context)
    return context_list

class Find_k_sentence:
    def __init__(self):
        name='checkpoint'
        self.model = SentenceTransformer(name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=self.model.to(self.device)

    def find_top_k(self,top_k, model, question, corpus, corpus_embeddings):
        query_embedding = model.encode(question, convert_to_tensor=True).to(self.device)
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.cpu()
        if len(corpus)>top_k:
            top_results = torch.topk(cos_scores, k=top_k)
            sentence_new_context = []
            score_list=[]
            for score, idx in zip(top_results[0], top_results[1]):
                sentence_new_context.append(f"{drop_last_dot(corpus[idx])}.")
                score_list.append(score)
            return sentence_new_context,score_list
        else:
            top_results = torch.topk(cos_scores, k=len(corpus))
            sentence_new_context = []
            score_list=[]
            for score, idx in zip(top_results[0], top_results[1]):
                sentence_new_context.append(f"{drop_last_dot(corpus[idx])}.")
                score_list.append(score.item())
            return sentence_new_context,score_list
    
    def update_data(self, top_k, data):
        new_context=[]
        new_question=[]
        new_answer=[]
        new_start=[]
        new_end=[]
        label=[]
        all_idx=[]
        score_list=[]
        for it,i in enumerate(tqdm(range(len(data)))):
            idx=data['idx'][i]
            context = data['context'][i]
            question = preprocess_text(data['question'][i])
            answer = preprocess_text(data['answer'][i])
            start_answer = data['start'][i]
            end_answer= data['end'][i]
            label.append(data['label'][i])
            corpus = split_sentence(context)

            corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True).to(self.device)
            sentence_new_context,score = self.find_top_k(top_k, self.model, question, corpus, corpus_embeddings) 
            # context_ = ' '.join([pa for pa in corpus if pa in sentence_new_context])
            context=' '.join(sentence_new_context)
            if answer in context:
                start_answer = context.find(answer)
                end_answer  = start_answer + len(answer) - 1
            else:
                start_answer=len(context)-1
                end_answer=len(context)-1
            
            score_list.append(score)
            new_context.append(context)
            new_question.append(question)
            new_answer.append(answer)
            new_start.append(start_answer)
            new_end.append(end_answer)
            all_idx.append(idx)

        new_data={'idx':all_idx,
                'context':new_context,
                'question':new_question,
                'answer':new_answer,
                'start':new_start,
                'end': new_end,
                'label':label,
                'score':score_list,}
        df=pd.DataFrame(new_data)
        return df

    def update_data_test(self,top_k, data):
        new_context=[]
        new_question=[]
        all_idx=[]
        score_list=[]
        for it,i in enumerate(tqdm(range(len(data)))):
            idx=data['idx'][i]
            context = data['context'][i]
            question = preprocess_text(data['question'][i])

            corpus = split_sentence(context)
            corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True).to(self.device)
            sentence_new_context,score = self.find_top_k(top_k, self.model, question, corpus, corpus_embeddings) 
            # context_ = ' '.join([pa for pa in corpus if pa in sentence_new_context])
            context=' '.join(sentence_new_context)
            score_list.append(score)
            new_context.append(context)
            new_question.append(question)
            all_idx.append(idx)

        new_data={'idx':all_idx,'context':new_context,
                'question':new_question,'score':score_list,}
        df=pd.DataFrame(new_data)
        return df

def main():
    find_k=Find_k_sentence()
    data=pd.read_csv('./data/data.csv')
    df=find_k.update_data(top_k=1,data=data)
    df.to_csv('./data/data_new.csv',index=False)

if __name__ == '__main__':
    main()