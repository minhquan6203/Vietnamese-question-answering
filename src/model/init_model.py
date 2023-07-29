from model.svm_model import createSVM_Model,SVM_Model
from model.transformer_model import createTrans_Model,Trans_Model
from model.transformer_svm_model import createTrans_SVM_Model,Trans_SVM_Model
from model.bert_cnn_model import createText_CNN_Model,Text_CNN_Model
from model.pat_model import createParallelAttentionTransformer,ParallelAttentionTransformer
from model.transformer_model_new import createTrans_Model_New,Trans_Model_New
from model.pair_sentence import createPair_Sentence_Model,Pair_Sentence_Model
from model.t5_model import createT5_Model,T5_Model

def build_model(config, answer_space):
    if config['model']['type_model']=='svm':
        return createSVM_Model(config, answer_space)
    if config['model']['type_model']=='trans':
        return createTrans_Model(config, answer_space)
    if config['model']['type_model']=='trans_new':
        return createTrans_Model_New(config, answer_space)
    if config['model']['type_model']=='trans_svm':
        return createTrans_SVM_Model(config, answer_space)
    if config['model']['type_model']=='cnn':
        return createText_CNN_Model(config, answer_space)
    if config['model']['type_model']=='pat':
        return createParallelAttentionTransformer(config,answer_space)
    if config['model']['type_model']=='pair_sentence':
        return createPair_Sentence_Model(config,answer_space)
    if config['model']['type_model']=='t5':
        return createT5_Model(config,answer_space)
    
def get_model(config, num_labels):
    if config['model']['type_model']=='svm':
        return SVM_Model(config, num_labels)
    if config['model']['type_model']=='trans':
        return Trans_Model(config, num_labels)
    if config['model']['type_model']=='trans_new':
        return Trans_Model_New(config,num_labels)
    if config['model']['type_model']=='trans_svm':
        return Trans_SVM_Model(config, num_labels)
    if config['model']['type_model']=='cnn':
        return Text_CNN_Model(config, num_labels)
    if config['model']['type_model']=='pat':
        return ParallelAttentionTransformer(config, num_labels)
    if config['model']['type_model']=='pair_sentence':
        return Pair_Sentence_Model(config, num_labels)
    if config['model']['type_model']=='t5':
        return T5_Model(config, num_labels)