import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_utils.load_data_longformer import Longformer_Loader
from model.longformer_model import Longformer_Model
from eval_metric.evaluate import ScoreCalculator
from tqdm import tqdm

class Longformer_Task:
    def __init__(self, config):
        self.num_epochs = config['train']['num_train_epochs']
        self.patience = config['train']['patience']
        self.learning_rate = config['train']['learning_rate']
        self.save_path = config['train']['output_dir']
        self.best_metric= config['train']['metric_for_best_model']
        self.pretraining=config['train']['pretraining']
        self.dataloader = Longformer_Loader(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model=Longformer_Model(config).to(self.device)
        self.compute_score = ScoreCalculator()
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler()
        lambda1 = lambda epoch: 0.95 ** epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
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
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    start_logits, end_logits, loss = self.base_model(question, context, start_idx, end_idx ,answers)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                train_loss += loss
            self.scheduler.step()
            train_loss /=len(train)

            with torch.no_grad():
                for it, (question, context, start_idx, end_idx ,answers, id) in enumerate(tqdm(valid)):
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
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
