from model.t5_model import createT5_Model,T5_Model

def build_model(config):
    if config['model']['type_model']=='t5':
        return createT5_Model(config)
    
def get_model(config):
    if config['model']['type_model']=='t5':
        return T5_Model(config)