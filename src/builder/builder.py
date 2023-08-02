from task.train_bert import Bert_Task
from task.train_t5 import T5_Task
from task.train_bart import Bart_Task
from task.train_roberta import Roberta_Task
from task.train_gpt2 import Gpt2_Task
from task.train_longformer import Longformer_Task

from task.inference_bert import Bert_Predict
from task.inference_t5 import T5_Predict
from task.inference_bart import Bart_Predict
from task.inference_roberta import Roberta_Predict
from task.inference_gpt2 import Gpt2_Predict
from task.inference_longformer import Longformer_Predict

def build_task(config):
    if config['model']['type_model']=='t5':
        return T5_Task(config), T5_Predict(config)
    if config['model']['type_model']=='bert':
        return Bert_Task(config), Bert_Predict(config)
    if config['model']['type_model']=='bart':
        return Bart_Task(config), Bart_Predict(config)
    if config['model']['type_model']=='roberta':
        return Roberta_Task(config), Roberta_Predict(config)
    if config['model']['type_model']=='gpt2':
        return Gpt2_Task(config), Gpt2_Predict(config)
    if config['model']['type_model']=='longformer':
        return Longformer_Task(config), Longformer_Predict(config)
    