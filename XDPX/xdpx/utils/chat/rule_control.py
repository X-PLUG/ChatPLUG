from dataclasses import dataclass
from typing import List
import json
from xdpx.utils import io
import re
from xdpx.utils.chat import regex_group_match
from xdpx.utils import download_from_url


class Rule:
    def __init__(self, name: str, pos_regex_list: List[str], neg_regex_list: List[str], response: List[str],
                 append_knowledge: str, reform_query: str, invoke_skill: bool):
        self.name = name
        self.pos_regex_list = [re.compile(p) for p in pos_regex_list] if pos_regex_list is not None else []
        self.neg_regex_list = [re.compile(p) for p in neg_regex_list] if neg_regex_list is not None else []
        self.response = response
        self.append_knowledge = append_knowledge
        self.reform_query = reform_query
        self.invoke_skill = bool(invoke_skill)

    def hit(self, query):
        if regex_group_match(query, self.pos_regex_list) and not regex_group_match(query, self.neg_regex_list):
            return True
        else:
            return False

    def asdict(self):
        result = {
            'name': self.name,
            'response': self.response,
            'append_knowledge': self.append_knowledge,
            'reform_query': self.reform_query,
            'invoke_skill': self.invoke_skill
        }
        return result


class RuleControl(object):
    def __init__(self, path):
        rules = []
        # for deploying on public cloud by aquila (can not read objects from different regions)
        if str(path).startswith('http'):
            path = download_from_url(path)
        rules_data = json.load(io.open(path))
        for rule in rules_data:
            rules.append(Rule(
                name=rule.get('name'),
                pos_regex_list=rule.get('pos_regex_list'),
                neg_regex_list=rule.get('neg_regex_list'),
                response=rule.get('response'),
                append_knowledge=rule.get('append_knowledge'),
                reform_query=rule.get('reform_query'),
                invoke_skill=rule.get('invoke_skill')
            ))
        self.rules = rules

    def call(self, query):
        for rule in self.rules:
            if rule.hit(query):
                return rule
        return None


if __name__ == '__main__':
    # control = RuleControl("oss://xdp-expriment/jiayi.qm/0_digital_human/configs/rules/rule_control_weather.json")
    control = RuleControl("oss://xdp-expriment/gaoxing.gx/chat/rules/safety_rule.json")
    a = control.call("阿里股价多少")
    b = control.call("今天天气如何")
    print("ok")
    while True:
        sentence = input("输入一个规则：")
        result = control.call(sentence)
        if result:
            print('hit rule')
            print(result.invoke_skill)
            print(result.response)
        else:
            print(None)
