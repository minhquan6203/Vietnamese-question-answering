from task.train_bert import Bert_Task
from task.train_t5 import T5_Task
from task.inference_bert import Bert_Predict
from task.inference_t5 import T5_Predict


def build_task(config):
    if config['model']['type_model']=='t5':
        return T5_Task(config), T5_Predict(config)
    if config['model']['type_model']=='bert':
        return Bert_Task(config), Bert_Predict(config)
    