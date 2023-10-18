
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import os
import numpy as np
import math
import re

class Pretraining_Dataset(Dataset):
    def __init__(self,config ,print_text=False):
      self.input_length = config["tokenizer"]["max_input_length"]
      self.path = config['data']['pretraining_dataset']
      self.dataset = self.split_into_segment(pd.read_csv(self.path),self.input_length)
      self.print_text = print_text

    def split_into_segment(self, ds, input_length):
        all_rows = []
        for index, row in ds.iterrows():
            text=row['context'].split()
            if len(text)>=7 and len(text) <= input_length:
                all_rows.append(' '.join(text))
        
            if len(text) > input_length:
                num_splits = int(len(text) / input_length)
                for i in range(num_splits):
                    start_idx = i * input_length
                    end_idx = (i + 1) * input_length
                    if end_idx > len(text):
                        end_idx = len(text) - 1
                    
                    scene_part = text[start_idx:end_idx]
                    if len(scene_part) >= 7:
                        all_rows.append(' '.join(scene_part))   
        ds2 = pd.DataFrame({'context':all_rows})
        return ds2

    def __len__(self):
        return len(self.dataset)
    
    def clean_text(self, text):
        text = re.sub(r"['\",\.\?:\-!]", "", text)
        text = text.strip()
        text = " ".join(text.split())
        text = text.lower()
        return text

    def span_corruption_mask(self, text, noise_span_length=3, noise_density=.2):
        max_index = len(text.split())
        mask = max_index * [0]
        span_num = math.ceil(( max_index * noise_density ) / 3 )
        exclude=[max_index-2, max_index-1]
        for i in range(span_num):
            while True:
                rand_num = np.random.randint(low=0, high=max_index) #Getting random number for mask index
                if rand_num not in exclude:
                    span = [rand_num, rand_num+1, rand_num+2]
                    for s in span:
                        mask[s] = 1
                        exclude.append(s)
                    if rand_num==1:
                        exclude.append(rand_num-1)
                    elif rand_num==2:
                        exclude.append(rand_num-1)
                        exclude.append(rand_num-2)
                    elif rand_num>2:
                        exclude.append(rand_num-1)
                        exclude.append(rand_num-2)
                        exclude.append(rand_num-3)
                    if not rand_num==max_index-3:
                        exclude.append(span[-1]+1)
                    break
                else:
                    continue
        return mask
    
    def noise_span_to_unique_sentinel(self, text, mask,sentinels):
        tokens = text.split()
        text_ = []
        one_count=0
        sentinel_cnt=0
        for i in range(len(tokens)):
            if mask[i] == 1:
                one_count+=1
                if one_count==1:
                    text_.append(sentinels[sentinel_cnt])
                    sentinel_cnt+=1
                else:
                    if one_count==3:
                        one_count=0
            else:
                text_.append(tokens[i])
        text_ = ' '.join(text_)
        return text_

    def nonnoise_span_to_unique_sentinel(self, text, mask,sentinels):
        tokens = text.split()
        text_ = []
        zero_first=True
        sentinel_cnt=0
        for i in range(len(tokens)):
            if mask[i] == 0:
                if zero_first:
                    text_.append(sentinels[sentinel_cnt])
                    zero_first=False
                    sentinel_cnt+=1
            else:
                zero_first=True
                text_.append(tokens[i])
        text_ = ' '.join(text_)
        return text_

    def convert_to_features(self, example_batch):  
        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch['context']))
        text = self.clean_text(example_batch['context'])
        mask = self.span_corruption_mask(text)
        sentinels=[]
        for i in range(100):
            sentinels.append(f'<extra_id_{i}>')
        input_ = self.noise_span_to_unique_sentinel(text,mask,sentinels)
        target_ = self.nonnoise_span_to_unique_sentinel(text,mask,sentinels)
        return input_, target_
  
    def __getitem__(self, index):
        id=1
        input_text, target = self.convert_to_features(self.dataset.iloc[index])
        return input_text, target, id


class T5_Pretraining_Loader:
    def __init__(self, config):
        self.train_batch=config['train']['per_device_train_batch_size']
        self.val_batch=config['train']['per_device_valid_batch_size']
        self.data = Pretraining_Dataset(config)
    def load_train_dev(self):
        dataset_size = len(self.data)
        train_size = int(0.9 * dataset_size)  # You can adjust the split ratio as needed
        dev_size = dataset_size - train_size
        train_set, val_set = random_split(self.data, [train_size, dev_size])
        
        train_loader = DataLoader(train_set, batch_size=self.train_batch, num_workers=2,shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.val_batch, num_workers=2,shuffle=True)
        return train_loader, val_loader