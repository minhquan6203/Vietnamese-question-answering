from task.train.train_bert import Bert_Task
from task.train.train_t5 import T5_Task
from task.train.train_bart import Bart_Task
from task.train.train_roberta import RoBerta_Task
from task.train.train_gpt2 import Gpt2_Task
from task.train.train_longformer import Longformer_Task

from task.infer.inference_bert import Bert_Predict
from task.infer.inference_t5 import T5_Predict
from task.infer.inference_bart import Bart_Predict
from task.infer.inference_roberta import Roberta_Predict
from task.infer.inference_gpt2 import Gpt2_Predict
from task.infer.inference_longformer import Longformer_Predict

def build_task(config):
    if config['model']['type_model']=='t5':
        return T5_Task(config), T5_Predict(config)
    if config['model']['type_model']=='bert':
        return Bert_Task(config), Bert_Predict(config)
    if config['model']['type_model']=='bart':
        return Bart_Task(config), Bart_Predict(config)
    if config['model']['type_model']=='roberta':
        return RoBerta_Task(config), Roberta_Predict(config)
    if config['model']['type_model']=='gpt2':
        return Gpt2_Task(config), Gpt2_Predict(config)
    if config['model']['type_model']=='longformer':
        return Longformer_Task(config), Longformer_Predict(config)
    