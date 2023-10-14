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

# model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
# model = SentenceTransformer('intfloat/multilingual-e5-large')
# model = SentenceTransformer('keepitreal/vietnamese-sbert')
def normalize_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower().strip()
    return text

def drop_last_dot(sentence):
    if sentence[-1] == '.':
        sentence = sentence[:-1]
    return sentence

def get_tokenizer(tokenizer):
    if callable(tokenizer):
        return tokenizer
    elif tokenizer is None:
        return lambda s: s 
    elif tokenizer == "pyvi":
        try:
            from pyvi import ViTokenizer
            return ViTokenizer.tokenize
        except ImportError:
            print("Please install PyVi package. "
                  "See the docs at https://github.com/trungtv/pyvi for more information.")
    elif tokenizer == "spacy":
        try:
            from spacy.lang.vi import Vietnamese
            return Vietnamese()
        except ImportError:
            print("Please install SpaCy and the SpaCy Vietnamese tokenizer. "
                  "See the docs at https://gitlab.com/trungtv/vi_spacy for more information.")
            raise
        except AttributeError:
            print("Please install SpaCy and the SpaCy Vietnamese tokenizer. "
                  "See the docs at https://gitlab.com/trungtv/vi_spacy for more information.")
            raise
    elif tokenizer == "vncorenlp":
        try:
            from vncorenlp import VnCoreNLP
            # before using vncorenlp, please run this command in your terminal:
            # vncorenlp -Xmx500m data_utils/vncorenlp/VnCoreNLP-1.1.1.jar -p 9000 -annotators wseg &
            annotator = VnCoreNLP(address="http://127.0.0.1", port=9000, max_heap_size='-Xmx500m')

            def tokenize(s: str):
                words = annotator.tokenize(s)[0]
                return " ".join(words)

            return tokenize
        except ImportError:
            print("Please install VnCoreNLP package. "
                  "See the docs at https://github.com/vncorenlp/VnCoreNLP for more information.")
            raise
        except AttributeError:
            print("Please install VnCoreNLP package. "
                  "See the docs at https://github.com/vncorenlp/VnCoreNLP for more information.")
            raise

def preprocess_sentence(sentence: str, tokenizer: str=None):
    sentence = sentence.lower()
    sentence = unicodedata.normalize('NFC', sentence)
    sentence = re.sub(r"[“”]", "\"", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\?", " ? ", sentence)
    sentence = re.sub(r":", " : ", sentence)
    sentence = re.sub(r";", " ; ", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"\"", " \" ", sentence)
    sentence = re.sub(r"'", " ' ", sentence)
    sentence = re.sub(r"\(", " ( ", sentence)
    sentence = re.sub(r"\[", " [ ", sentence)
    sentence = re.sub(r"\)", " ) ", sentence)
    sentence = re.sub(r"\]", " ] ", sentence)
    sentence = re.sub(r"/", " / ", sentence)
    sentence = re.sub(r"\.", " . ", sentence)
    sentence = re.sub(r"-", " - ", sentence)
    sentence = re.sub(r"\$", " $ ", sentence)
    sentence = re.sub(r"\&", " & ", sentence)
    sentence = re.sub(r"\*", " * ", sentence)
    # tokenize the sentence
    tokenizer = get_tokenizer(tokenizer)
    sentence = tokenizer(sentence)
    sentence = " ".join(sentence.strip().split()) # remove duplicated spaces
    # tokens = sentence.strip().split()
    
    return sentence

def split_sentence(paragraph):
    paragraph=paragraph.replace(':\n\n',': ').strip()
    pre_context_list = paragraph.split("\n\n")
    context_list = []
    for context in pre_context_list:
        sen = context.split(". ")
        context_list = context_list + sen
    return context_list

# def preprocess_ev(claim,ev):
#     if len(ev.split())<5:
#         ev=claim
#     else:
#         ev=' '.join(split_sentence(ev))
#     return ev

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
                sentence_new_context.append(f"{drop_last_dot(corpus[idx])}.")
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
            answer = ' '.join(split_sentence(data['answer'][i]))
            start_answer = data['start'][i]
            end_answer= data['end'][i]

            label.append(data['label'][i])
            corpus = split_sentence(context)
            ques_list = split_sentence(question)
            corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True).to(self.device)
            multi_context=[]
            for ques in ques_list:
                sentence_new_context = self.find_top_k(top_k, self.model, ques, corpus, corpus_embeddings) 
                # context_ = ' '.join([pa for pa in corpus if pa in sentence_new_context])
                for c in sentence_new_context:
                    if c not in multi_context:
                        multi_context.append(c)
            context=' '.join(multi_context)
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
            ques_list = split_sentence(question)
            corpus_embeddings = self.model.encode(corpus, convert_to_tensor=True).to(self.device)
            multi_context=[]
            for ques in ques_list:
                sentence_new_context = self.find_top_k(top_k, self.model, ques, corpus, corpus_embeddings) 
                # context_ = ' '.join([pa for pa in corpus if pa in sentence_new_context])
                for c in sentence_new_context:
                    if c not in multi_context:
                        multi_context.append(c)
            context=' '.join(multi_context)

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