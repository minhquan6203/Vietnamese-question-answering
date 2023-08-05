import numpy as np
import string
class F1:
  def Precision(self,y_true,y_pred):
    common = set(y_true) & set(y_pred)
    return len(common) / len(set(y_pred))

  def Recall(self,y_true,y_pred):
    common = set(y_true) & set(y_pred)
    return len(common) / len(set(y_true))

  def Compute(self,y_true,y_pred):
    if len(y_pred) == 0 or len(y_true) == 0:
        return int(y_pred == y_true)

    precision = self.Precision(y_true, y_pred)
    recall = self.Recall(y_true, y_pred)

    if precision == 0 or recall == 0:
        return 0
    f1 = 2*precision*recall / (precision+recall)
    return f1

def normalize_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    return text

class ScoreCalculator:
    
    def f1_token(self, pred_tokens, answers) -> float:
        #F1 score token level
        pred_tokens=normalize_text(pred_tokens)
        answers=normalize_text(answers)
        f1=F1()
        scores=[]
        for i in range(len(answers)):
            scores.append(f1.Compute(answers[i].split(),pred_tokens[i].split()))
        return np.mean(scores)
    
    def f1_char(self, pred_tokens, answers) -> float:
        #F1 score char level
        pred_tokens=normalize_text(pred_tokens)
        answers=normalize_text(answers)
        f1=F1()
        scores=[]
        for i in range(len(answers)):
            scores.append(f1.Compute(answers[i],pred_tokens[i]))
        return np.mean(scores)

    def exact_macth(self, pred_tokens, answers) -> float:
        pred_tokens=normalize_text(pred_tokens)
        answers=normalize_text(answers)
        scores=[]
        for i in range(len(answers)):
            if answers[i]==pred_tokens[i]:
                scores.append(1)
            else:
                scores.append(0)
        return np.mean(scores)
