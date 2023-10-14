from sentence_transformers import util
import torch
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from underthesea import sent_tokenize
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
# model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
# model = SentenceTransformer('intfloat/multilingual-e5-large')
# model = SentenceTransformer('keepitreal/vietnamese-sbert')

def split_sentence(paragraph):
    pre_context_list = paragraph.split("\n\n")
    context_list = []
    for context in pre_context_list:
        sen = context.split(". ")
        context_list.append(sen)
    return context_list

class Find_k_sentence:
    def __init__(self):
        name='keepitreal/vietnamese-sbert'
        self.model = SentenceTransformer(name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=self.model.to(self.device)

    def find_top_k(self,top_k, model, question, corpus, corpus_embeddings):
        if len(corpus)>top_k:
            query_embedding = model.encode(question, convert_to_tensor=True).to(self.device)
            cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
            cos_scores = cos_scores.cpu()
            top_results = torch.topk(cos_scores, k=top_k)
            sentence_new_context = []
            for score, idx in zip(top_results[0], top_results[1]):
                sentence_new_context.append(corpus[idx])
            return sentence_new_context
        else:
            return corpus
    
    def update_data(self, top_k, data):
        new_context=[]
        new_question=[]
        new_answer=[]
        new_start=[]
        new_end=[]
        label=[]
        all_idx=[]
        for it,i in enumerate(tqdm(range(len(data)))):
            idx=data['idx'][i]
            context = data['context'][i]
            question = data['question'][i]
            answer = data['answer'][i]
            start_answer = 0
            end_answer= 0

            label.append(data['label'][i])
            corpus = split_sentence(context)
            corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True).to(self.device)
            sentence_new_context = self.find_top_k(top_k, self.model, question, corpus, corpus_embeddings) 
            # context_ = ' '.join([pa for pa in corpus if pa in sentence_new_context])
            context=' '.join(sentence_new_context)
            if answer in context:
                start_answer = context.find(answer)
                end_answer  = start_answer + len(answer) - 1
            else:
                start_answer = 0
                end_answer = 0

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
                'label':label}
        df=pd.DataFrame(new_data)
        return df

    def update_data_test(self,top_k, data):
        new_context=[]
        new_question=[]
        all_idx=[]
        for it,i in enumerate(tqdm(range(len(data)))):
            idx=data['idx'][i]
            context = data['context'][i]
            question = data['question'][i]

            corpus = split_sentence(context)
            corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True).to(self.device)
            sentence_new_context = self.find_top_k(top_k, self.model, question, corpus, corpus_embeddings) 
            # context_ = ' '.join([pa for pa in corpus if pa in sentence_new_context])
            context=' '.join(sentence_new_context)

            new_context.append(context)
            new_question.append(question)
            all_idx.append(idx)

        new_data={'idx':all_idx,'context':new_context,
                'question':new_question}
        df=pd.DataFrame(new_data)
        return df

def main():
    find_k=Find_k_sentence()
    data=pd.read_csv('./data/data.csv')
    df=find_k.update_data(top_k=5,data=data)
    df.to_csv('./data/data_new.csv')

if __name__ == '__main__':
    main()