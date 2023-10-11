import os
import logging
from typing import Text, Dict, List
import pandas as pd
import torch
import transformers
from data_utils.load_data_bert import Bert_Loader
from model.bert_model import Bert_Model
from tqdm import tqdm

class Bert_Predict:
    def __init__(self, config: Dict):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path=os.path.join(config["train"]["output_dir"], "best_model.pth")
        self.model = Bert_Model(config)
        self.dataloader = Bert_Loader(config)
    def predict_submission(self):
        transformers.logging.set_verbosity_error()
        logging.basicConfig(level=logging.INFO)
    
        # Load the model
        logging.info("Loading the {0} model...".format(self.model_name))
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        # Obtain the prediction from the model
        logging.info("Obtaining predictions...")
        test =self.dataloader.load_test()
        submits=[]
        ids=[]
        self.model.eval()
        with torch.no_grad():
            for it, (question, context, id) in enumerate(tqdm(test)):
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    pred_tokens = self.model(question, context)
                    submits.extend(pred_tokens)
                    ids.extend(id)

        data = {'id': ids,'label': submits }
        df = pd.DataFrame(data)
        df.to_csv('./submission.csv', index=False)
        