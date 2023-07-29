from text_module.t5_embedding import T5_tokenizer
import numpy as np
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

class ScoreCalculator:
    def __init__(self,config):
        self.tokenizer=T5_tokenizer(config)
        self.padding = config["tokenizer"]["padding"]
        self.max_input_length = config["tokenizer"]["max_input_length"]
        self.max_target_length = config["tokenizer"]["max_target_length"]
        self.truncation = config["tokenizer"]["truncation"]

    def f1_token(self, model, contexts, questions, answers) -> float:
        encoded_inputs = self.tokenizer(
                        contexts,questions,
                        padding= self.padding,
                        max_length=self.max_input_length,
                        truncation=self.truncation,
                        return_tensors='pt',
                    )
        preds_ids=model.generate(**encoded_inputs)
        preds=self.tokenizer.batch_decode(preds_ids, skip_special_tokens=True)
        #F1 score token level
        f1=F1()
        scores=[]
        for i in range(len(answers)):
            scores.append(f1.Compute(answers[i].split(),preds[i].split()))
        return np.mean(scores)
