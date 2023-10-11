import os
import logging
from typing import Text, Dict, List
import pandas as pd
import torch
import transformers
from data_utils.load_data_t5 import T5_Loader
from model.t5_model import T5_Model
from tqdm import tqdm

class T5_Predict:
    def __init__(self, config: Dict):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path=os.path.join(config["train"]["output_dir"], "best_model.pth")
        self.model = T5_Model(config)
        self.dataloader = T5_Loader(config)
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
        

