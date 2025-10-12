from typing import List
from main.utils.label_tokenizer import LabelTokenizer


class SpanNERMetrics():
    
    def __init__(self, labelTokenizer: LabelTokenizer) -> None:
        self.labelTokenizer = labelTokenizer
        self.reset()
        self.f1_metrics = F1Metrics()
        
    def reset(self):
        self.matches_numbers = [0] * len(self.labelTokenizer)
        self.gold_numbers = [0] * len(self.labelTokenizer)
        self.pred_numbers = [0] * len(self.labelTokenizer)
    
    def add(self, pred_entities_list:List[set], gold_entities_list: List[set]):
        # 按照类别来计算，并且默认第一位是O
        for i in range(1, len(self.labelTokenizer)):
            for pred_entities, gold_entities in zip(pred_entities_list, gold_entities_list):
                # {{start, end, label_idx}} => {start, end, label_idx}
                pred_type_entities = set(j for j in pred_entities if j[-1]==i)
                gold_type_entities = set(j for j in gold_entities if j[-1]==i)
                matches_number, pred_number, gold_number = len(pred_type_entities & gold_type_entities),len(pred_type_entities), len(gold_type_entities)
                self.matches_numbers[i] += matches_number
                self.pred_numbers[i] += pred_number
                self.gold_numbers[i] += gold_number
                # 计算总的数量
                self.matches_numbers[0] += matches_number
                self.pred_numbers[0] += pred_number
                self.gold_numbers[0] += gold_number
                
    def compute(self):
        reports = [None] * len(self.labelTokenizer)
        reports_dict = {}
        for i in range(len(reports)):
            reports[i] = self.f1_metrics(
                self.matches_numbers[i],
                self.pred_numbers[i],
                self.gold_numbers[i]
            )
            reports[i]["supports"] = self.gold_numbers[i]
            if i==0:
                for key in reports[i]:
                    reports_dict[key] = reports[i][key]
            else:
                prefix = self.labelTokenizer.convert_ids_to_tokens(i)
                for key in reports[i]:
                    reports_dict[f"{prefix}_{key}"] = reports[i][key]
        return reports_dict

class F1Metrics():
    
    def __init__(self) -> None:
        pass
    
    def precision(self, matches: int, predition_number: int):
        return matches/predition_number if predition_number>0 else 0
    
    def recall(self, matches: int, gold_number: int):
        return matches/gold_number if gold_number>0 else 0
    
    def f1_score(self, matches: int, predition_number: int, gold_number: int):
        precision = self.precision(matches, predition_number)
        recall = self.recall(matches, gold_number)
        return (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall != 0)
            else 0.0
        )
        
    def __call__(self, matches: int, predition_number: int, gold_number: int) -> dict:
        return {
            "f1": self.f1_score(matches, predition_number, gold_number),
            "precision": self.precision(matches, predition_number),
            "recall": self.recall(matches, gold_number)
        }
        
        