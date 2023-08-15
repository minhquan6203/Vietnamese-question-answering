from sentence_transformers import util
import torch
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from underthesea import sent_tokenize
model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')

def find_top_k(top_k, model, question, corpus, corpus_embeddings):
    if len(corpus)>top_k:
        query_embedding = model.encode(question, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        cos_scores = cos_scores.cpu()

        top_results = torch.topk(cos_scores, k=top_k)


        sentence_new_context = []
        for score, idx in zip(top_results[0], top_results[1]):
            sentence_new_context.append(corpus[idx])
        
        return sentence_new_context
    else:
        return corpus
  

def update_data(top_k, data):
    new_context=[]
    new_question=[]
    new_answer=[]
    new_start=[]
    new_end=[]
    label=[]
    for i in range(len(data)):
        id=1
        context = data['context'][i]
        question = data['question'][i]
        answer = data['answer'][i]
        start_answer = data['start'][i]
        end_answer= data['end'][i]

        label.append(data['label'][i])
        corpus = sent_tokenize(context)
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

        if len(corpus)>top_k:
            sentence_new_context = find_top_k(top_k, model, question, corpus, corpus_embeddings) 
            context_ = ' '.join([pa for pa in corpus if pa in sentence_new_context])
            if answer in context_:
                context = context_
                start_answer = context.find(answer)
                end_answer  = start_answer + len(answer)

        new_context.append(context)
        new_question.append(question)
        new_answer.append(answer)
        new_start.append(start_answer)
        new_end.append(end_answer)

    new_data={'context':new_context,
            'question':new_question,
            'answer':new_answer,
            'start':new_start,
            'end': new_end,
            'label':label}
    df=pd.DataFrame(new_data)
    return df

def update_data_test(top_k, data):
    new_context=[]
    new_question=[]
    for i in range(len(data)):
        id=1
        context = data['context'][i]
        question = data['question'][i]

        corpus = sent_tokenize(context)
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    
        if len(corpus)>top_k:
            sentence_new_context = find_top_k(top_k, model, question, corpus, corpus_embeddings) 
            context_ = ' '.join([pa for pa in corpus if pa in sentence_new_context])
            context = context_

        new_context.append(context)
        new_question.append(question)

    new_data={'context':new_context,
            'question':new_question}
    df=pd.DataFrame(new_data)
    return df

def main():
    data=pd.read_csv('/content/drive/MyDrive/QA/QA.csv')
    df=update_data(top_k=5,data=data)
    df.to_csv('QA_new.csv')

if __name__ == '__main__':
    main()