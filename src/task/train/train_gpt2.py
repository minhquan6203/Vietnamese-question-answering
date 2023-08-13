import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_utils.load_data_gpt2 import Gpt2_Loader
from model.gpt2_model import Gpt2_Model
from eval_metric.evaluate import ScoreCalculator
from tqdm import tqdm

class Gpt2_Task:
    def __init__(self, config):
        self.num_epochs = config['train']['num_train_epochs']
        self.patience = config['train']['patience']
        self.learning_rate = config['train']['learning_rate']
        self.save_path = config['train']['output_dir']
        self.best_metric= config['train']['metric_for_best_model']
        self.dataloader = Gpt2_Loader(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model=Gpt2_Model(config).to(self.device)
        self.compute_score = ScoreCalculator()
        self.optimizer = optim.AdamW(self.base_model.parameters(), lr=self.learning_rate)
    def training(self):
        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)
    
        train, valid = self.dataloader.load_train_dev()

        if os.path.exists(os.path.join(self.save_path, 'last_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'last_model.pth'))
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['self.optimizer_state_dict'])
            print('loaded the last saved model!!!')
            initial_epoch = checkpoint['epoch'] + 1
            print(f"continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("first time training!!!")

        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'))
            best_score = checkpoint['score']
        else:
            best_score = 0.
            
        threshold=0
        self.base_model.train()
        for epoch in range(initial_epoch, self.num_epochs + initial_epoch):
            valid_f1 = 0.
            valid_em = 0.
            train_loss = 0.
            valid_loss = 0.
            for it, (question, context, start_idx, end_idx ,answers, id) in enumerate(tqdm(train)):
                self.optimizer.zero_grad()
                start_logits, end_logits, loss = self.base_model(question, context, start_idx, end_idx ,answers)
                loss.backward()
                self.optimizer.step()
                train_loss += loss
            train_loss /=len(train)

            with torch.no_grad():
                for it, (question, context, start_idx, end_idx ,answers, id) in enumerate(tqdm(valid)):
                    self.optimizer.zero_grad()
                    start_logits, end_logits, loss = self.base_model(question, context, start_idx, end_idx ,answers)
                    valid_loss += loss
                    pred_tokens = self.base_model(question, context, start_idx, end_idx)
                    valid_f1+=self.compute_score.f1_token(pred_tokens, answers)
                    valid_em+=self.compute_score.exact_match(pred_tokens, answers)
            valid_loss /=len(valid)
            valid_f1 /= len(valid)
            valid_em /=len(valid)

            print(f"epoch {epoch + 1}/{self.num_epochs + initial_epoch}")
            print(f"train loss: {train_loss:.4f}")
            print(f"valid loss: {valid_loss:.4f} valid em: {valid_em:.4f} valid f1_token: {valid_f1:.4f}")

            if self.best_metric=='f1':
                score=valid_f1
            if self.best_metric=='em':
                score=valid_em
            # save the last model
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.base_model.state_dict(),
                'self.optimizer_state_dict': self.optimizer.state_dict(),
                'score': score}, os.path.join(self.save_path, 'last_model.pth'))
            
            # save the best model
            if epoch > 0 and score < best_score:
              threshold += 1
            else:
              threshold = 0

            if score > best_score:
                best_score = score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.base_model.state_dict(),
                    'self.optimizer_state_dict': self.optimizer.state_dict(),
                    'score':score}, os.path.join(self.save_path, 'best_model.pth'))
                print(f"saved the best model with {self.best_metric} of {score:.4f}")
            
            # early stopping
            if threshold >= self.patience:
                print(f"early stopping after epoch {epoch + 1}")
                break
