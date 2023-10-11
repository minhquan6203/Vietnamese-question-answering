import os
import logging
from typing import Text, Dict, List
import pandas as pd
import torch
import transformers
from data_utils.load_data_bart import Bart_Loader
from model.bart_model import Bart_Model
from tqdm import tqdm

class Bart_Predict:
    def __init__(self, config: Dict):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path=os.path.join(config["train"]["output_dir"], "best_model.pth")
        self.model = Bart_Model(config)
        self.dataloader = Bart_Loader(config)
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
            for it, (input_text, id) in enumerate(tqdm(test)):
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    pred_tokens = self.model(input_text)
                    submits.extend(pred_tokens)
                    ids.extend(id)

        data = {'id': ids,'label': submits }
        df = pd.DataFrame(data)
        df.to_csv('./submission.csv', index=False)
        

