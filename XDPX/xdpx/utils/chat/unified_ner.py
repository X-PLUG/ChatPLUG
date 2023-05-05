from typing import List

import requests
import json
from dataclasses import dataclass


@dataclass
class NamedEntity:
    entity: str
    label: str
    score: float


class NERTool:
    def __init__(self, ner_url, threshold=0.85):
        self.url = ner_url
        self.threshold = threshold
        self.header = {'Content-Type': 'application/json', 'Accept-Encoding': 'utf-8'}
        print(f'| Using NER client: {self.url}, threshold: {self.threshold}')

    def recognize(self, query) -> List[NamedEntity]:
        '''

        Args:
            query:

        Returns:
            list of entities

        '''
        param = {"text": query}
        res = requests.post(self.url, data=json.dumps(param), headers=self.header)
        results = []
        try:
            res_data = json.loads(res.text)
            result = res_data['data']
            ner_results = json.loads(result)
            target_labels = ["Person", "Work", "Location", "Game", "Software", "Medicine", "Food", "Website",
                             "Disease&Symptom"]  # 仅仅链接这几类实体
            if ner_results:
                for each_result in ner_results:
                    label = each_result["label"]
                    entity = each_result["span"]
                    score = float(each_result["score"])
                    if label in target_labels and score > self.threshold:  # 过滤掉其余类别和低置信度样例
                        results.append(NamedEntity(entity=entity, label=label, score=score))

                results = sorted(results, key=lambda x: target_labels.index(x.label))  # 更重要的实体更靠前
        except:
            print(f"| NER Error: " + res.text)
        return results
