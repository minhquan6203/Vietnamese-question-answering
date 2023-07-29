import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_utils.load_data import Get_Loader
from model.init_model import build_model
from eval_metric.evaluate import ScoreCalculator
from data_utils.load_data import create_ans_space
from tqdm import tqdm

class NLI_Task:
    def __init__(self, config):
        self.num_epochs = config['train']['num_train_epochs']
        self.patience = config['train']['patience']
        self.learning_rate = config['train']['learning_rate']
        self.save_path = config['train']['output_dir']
        self.best_metric= config['train']['metric_for_best_model']
        self.answer_space=create_ans_space(config)
        self.dataloader = Get_Loader(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model=build_model(config,self.answer_space).to(self.device)
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
            valid_acc = 0.
            valid_f1 =0.
            train_loss = 0.
            valid_loss = 0.
            for it, (sent1, sent2, labels, id) in enumerate(tqdm(train)):
                self.optimizer.zero_grad()
                labels=labels.to(self.device)
                logits, loss = self.base_model(sent1, sent2, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss
            train_loss /=len(train)

            with torch.no_grad():
                for it, (sent1, sent2, labels, id) in enumerate(tqdm(valid)):
                    self.optimizer.zero_grad()
                    labels=labels.to(self.device)
                    logits, loss = self.base_model(sent1, sent2, labels)
                    preds = torch.argmax(logits, dim=-1)
                    valid_loss += loss
                    valid_acc+=self.compute_score.acc(labels,preds)
                    valid_f1+=self.compute_score.f1(labels,preds)
                    
            valid_loss /=len(valid)
            valid_acc /= len(valid)
            valid_f1 /= len(valid)

            print(f"epoch {epoch + 1}/{self.num_epochs + initial_epoch}")
            print(f"train loss: {train_loss:.4f}")
            print(f"valid loss: {valid_loss:.4f} valid acc: {valid_acc:.4f} valid f1: {valid_f1:.4f}")

            if self.best_metric =='accuracy':
                score=valid_acc
            if self.best_metric=='f1':
                score=valid_f1

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
