import yaml
import argparse
import logging
from typing import Text
import transformers

from task.train_t5 import T5_Task
from task.inference_t5 import Predict

def main(config_path: Text) -> None:
    transformers.logging.set_verbosity_error()
    logging.basicConfig(level=logging.INFO)
    
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    if config['model']['type_model']=='t5':
        logging.info("training started...")
        T5_Task(config).training()
        logging.info("training complete")
        
        logging.info('now evaluate on test data...')
        Predict(config).predict_submission()
        logging.info('task done!!!')
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    
    main(args.config)