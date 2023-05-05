import re
import torch

question_regex_group = [
    re.compile(r'(？|\?|吗|呢|什么|怎么|怎样|咋|啥|如何|为什么|哪|几|谁|多少|多大|多高|是不是|有没有|是否|多久|可不可以|能不能|行不行)'),
    re.compile('(是).+(还是)')]
persona_regex_group = [
    re.compile('(我|你)+(是|是不是|也是|叫啥|叫什么|几岁|多少岁|毕业|今年|多大|哪里|经常|一般|平时|平常|谁|会|还会|工作|名字|姓名|小名|大名|全名|年龄|年纪|工作|职业|干什么|专业)+'),
    re.compile('(我的|你的)+(名字|姓名|昵称|名称|全名|大名|年纪|年龄|工作|职业|学校|宠物|猫|狗|爱好|大学)+'),
    re.compile('(我|你)+(的)*(父母|爸|妈|男朋友|女朋友|哥|姐|妹|弟|老公|老婆|孩子|女儿|儿子)+'),
    re.compile('(我|你)+(是)*(男的|女的|男生|女生|男孩|女孩|性别)+'),
    re.compile('(我|你)+(是|叫|来自|在)+.*(专业|工作|哪里|男|女|毕业|名字)+'),
    re.compile('(你)+(是).+(还是)'),
    re.compile('^(你)+(是|叫|的|在)+'),
    re.compile('(你)+(看过|做过|喜欢|吃过|听过|已经|最喜欢)'),
    re.compile('(我|你)+(老家|来自|哪里)'),
    re.compile('(我|你)+(大学|上学|大几)'),
]

bye_response_keywords = ['再见', '拜拜', '不跟你聊', '去忙', '先忙', '不聊', '下次聊', '再聊', '打扰你', '下次再']
bye_regex = re.compile('|'.join(bye_response_keywords))
special_skill_regex = re.compile('新闻|股价|疫情|天气|市值')

def regex_group_match(query, patterns):
    for r in patterns:
        if r.findall(query):
            return True
    return False

def text_is_bye(query):
    return bool(bye_regex.findall(query))


def text_is_question(query):
    return regex_group_match(query.replace(' ', ''), question_regex_group)


def is_persona_question(q):
    if text_is_question(q):
        return regex_group_match(q, persona_regex_group)
    return False


def is_special_skill(query):
    return bool(special_skill_regex.findall(query))


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


if __name__ == '__main__':
    ""
    r = is_persona_question("你今年多大？")
    print(r)