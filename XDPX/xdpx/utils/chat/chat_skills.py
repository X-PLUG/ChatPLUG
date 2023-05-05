import re
import json
import random
import requests
from lunar_python import Solar
from datetime import datetime
from alibabacloud_chatbot20220408.client import Client as Chatbot20220408Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_chatbot20220408 import models as chatbot_20220408_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient

from xdpx.utils.chat import regex_group_match
from xdpx.utils.chat.thirdparty.Time_NLP.TimeNormalizer import TimeNormalizer

def get_date(sentence):
    """
    解析句子当中的获取日期信息
    Return：
        句子当中存在日期表达式："2022-08-29"
        不存在日期表达式：None
    """

    tn = TimeNormalizer()
    res = tn.parse(target=sentence)
    if "error" in res or "timestamp" not in res:
        return None
    time = json.loads(res)["timestamp"].split(" ")[0]
    return time


class CalendarSkill(object):
    name = 'calendar'
    time_regex = re.compile('(北京|现在|当前|今日).*(几点|时间)')
    day_regex = re.compile('(今天|明天|后天|大后天|昨天|前天|大前天|今日|昨日|明日).*(什么|啥|几|哪)(时间|时候|号|日期|日子|天)')
    day_regex_2 = re.compile('(下周|周|这周|上周|上上周|下下周)([一二三四五六天日]).*(什么|啥|几|哪)(时间|时候|号|日期|日子|天)')
    week_regex = re.compile('(今天|明天|后天|大后天|昨天|前天|大前天|今日|昨日|明日).*(星期几|周几|礼拜几)')
    jiaqi_regex = re.compile('(元旦|春节|清明|劳动节|端午|中秋|国庆).*几号')

    day_shift_map = {
        '今天': 0,
        '今日': 0,
        '明天': 1,
        '后天': 2,
        '大后天': 3,
        '昨天': -1,
        '前天': -2,
        '大前天': -3,
        '昨日': -1,
        '明日': 1
    }
    week_shift_map = {
        '周': 0,
        '这周': 0,
        '下周': 1,
        '上周': -1,
        '下下周': 2,
        '上上周': -2
    }
    weekday_shift_map = {
        '一': 0,
        '二': 1,
        '三': 2,
        '四': 3,
        '五': 4,
        '六': 5,
        '七': 6,
        '日': 6,
        '天': 6
    }
    week_tips = {"0":{"一": ["今天是周一", "今天是周一，忙day", "周一，周末过太快了", "一扭脸就到周一了", "今天周一，还没玩够就周一了唉"],
                      "二": ["今天周二", "今天周二，怎么才周二啊", "周二！什么...今天才周二"],
                      "三": ["今天周三，本周过半", "今天是周三", "周三周三，一座大山", "今天是周三，我要鼓足干劲了", "今天才周三"],
                      "四": ["今天是周四", "今天星期四，周四也要继续努力哦", "终于周四喽", "周四，舍思day", "周四啦，再忍忍，还有一天了"],
                      "五": ["今天是周五，明天就放假了，yeah~~", "今天周五，总算到周五了", "终于周五啦，撒花❀❀❀❀", "兄弟姐妹们，周五啦", "周五周五，生龙活虎", "周五周五，烦恼全无"],
                      "六": ["今天是星期六", "周六，约不约?", "周六，干点什么好呢？", "今天周六，周末快乐吖"],
                      "日": ["今天礼拜天", "今天周日", "周日，明明什么都没做，周末就过完了", "今天星期天，伤day", " 周日，一个说过去就过去的周末"]},
                 "1":{"一": ["明天是周一","明天是周一，忙day","周一，今天周末最后一天了", "周一，还没玩够就周一了唉"],
                      "二": ["明天周二", "明天周二，怎么才周二啊", "周二！ 什么...明天才周二"],
                      "三": ["明天是周三", "明天是礼拜三", "周三周三，一座大山", "明天才周三"],
                      "四": ["明天是周四", "周四，舍思day", "明天才周四啦，再忍忍，还有2天就周末了"],
                      "五": ["明天周五，总算快到周五了", "周五周五，生龙活虎", "周五周五，烦恼全无"],
                      "六": ["明天星期六，放假了", "周六，约不约? ", "周六，干点什么好呢？"],
                      "日": ["明天礼拜天", "明天周日", "明天星期天，伤day", "周日，一个说过去就过去的周末"]}}
    tips_map = {'今天': "0",
                '今日': "0",
                '明天': "1",
                '明日': "1"}


    def invoke(self, query, dialog_state, instance_code):
        date_now = Solar.fromDate(datetime.now())
        if self.time_regex.findall(query):
            day = date_now
            response = f'现在北京时间是 {day.getYear()}年{day.getMonth()}月{day.getDay()}日{day.getHour()}点{day.getMinute()}分'
            return response
        days = self.day_regex.findall(query)
        if days and days[0][0] in self.day_shift_map:
            day_shift = self.day_shift_map[days[0][0]]
            day = date_now.next(day_shift)
            response = f'{days[0][0]}是{day.getYear()}年{day.getMonth()}月{day.getDay()}日'
            return response

        days = self.day_regex_2.findall(query)
        if days:
            week_shift = self.week_shift_map[days[0][0]]
            weekday_shift = self.weekday_shift_map[days[0][1]]
            now_weekday = self.weekday_shift_map[date_now.getWeekInChinese()]
            shift = week_shift * 7 + weekday_shift - now_weekday
            day = date_now.next(shift)
            response = f'{days[0][0]}{days[0][1]}是{day.getYear()}年{day.getMonth()}月{day.getDay()}日'
            return response

        days = self.week_regex.findall(query)
        if days and days[0][0] in self.day_shift_map:
            day_shift = self.day_shift_map[days[0][0]]
            day = date_now.next(day_shift)
            if days[0][0] in self.tips_map:
                response = random.choice(self.week_tips[self.tips_map[days[0][0]]][day.getWeekInChinese()])
            else:
                response = f'{days[0][0]}是周{day.getWeekInChinese()}'
            return response

        # 最后无法判定的时候，直接time-normalizer解析整个query（兜底策略）
        final_response = get_date(query)
        if final_response:
            try:
                year = final_response.split("-")[0]
                month = final_response.split("-")[1]
                day = final_response.split("-")[2]
                response = f'是{year}年{month}月{day}号哦~'
            except:
                response = None
            return response


class DSSkill(object):
    
    def __init__(self):
        """ TODO: from alibaba cloud"""
        pass

    def invoke(self, query, dialog_state, instance_code):
        return "DSSkill"

class NewsSkill(object):
    def __init__(self):
        pass

    def invoke(self, query, dialog_state, instance_code):
        return "NewsSkill"


class JokeSkill(object):
    def __init__(self):
        pass

    def invoke(self, query, dialog_state, instance_code):
        return "JokeSkill"


class PoemSkill(object):
    def __init__(self):
        pass

    def invoke(self, query, dialog_state, instance_code):
        return "PoemSkill"

class ChatSkills:
    chat_skills = {
        '$$$查天气$$$': DSSkill(),
        '$$$星座运势$$$': DSSkill(),
        '$$$查日期$$$': CalendarSkill(),
        '$$$讲笑话$$$': JokeSkill(),
        '$$$讲新闻$$$': NewsSkill(),
        '$$$藏头诗$$$': PoemSkill(),
    }
    ds_skills = {
        '$$$查天气$$$': DSSkill(),
        '$$$星座运势$$$': DSSkill(),
    }
    local_ds_script = {
        '$$$藏头诗$$$': ["给你个机会为难我，你可以随便说4个字",
                        "写点什么好呢？要不你说个成语吧，我围绕成语来写",
                        "要不你随便说几个字吧，我试试看（ps.不要超过8个字哈）",
                        "Emmm，写什么呢？要不你来定吧，你说几个字好了"]
    }
    local_ds_script_xiaoda = {
        '$$$藏头诗$$$': ["给你个机会为难我，你可以随便说4个字",
                        "写点什么好呢？要不你说个成语吧，我围绕成语来写",
                        "要不你随便说几个字吧，我试试看（ps.不要超过8个字哈）",
                        "Emmm，写什么呢？要不你来定吧，你说几个字好了",
                        "不知道要写点啥，你来说几个字吧，我试试[思考]",
                        "你可以随便说4个字，我来写 [自信][自信][自信]",
                        "写诗啊~ 可以啊~写什么呢[疑问]，你来说几个字好了",
                        "写诗好难的，我学了好久呢~你想写什么？你来说几个字吧",
                        "这样好不好，你说个词吧，我试试吧"]
    }

    ds_tolerance_turns = 5

    def call(self, query, query_origin, invoke_skill_name, dialog_state, instance_code):
        #  for two turn local ds
        if dialog_state.get('ds_hang_skill') is not None:
            invoke_skill_name = dialog_state['ds_hang_skill']
            dialog_state['ds_hang_skill'] = None
            query = query_origin  # 采用改写前的query作为输入

        if invoke_skill_name and invoke_skill_name in self.chat_skills:  # 精准触发
            skill_name = invoke_skill_name
            skill_response = self.chat_skills[invoke_skill_name].invoke(query, dialog_state, instance_code)
        else:
            skill_name = random.choice(list(self.ds_skills.keys()))
            chat_skill = self.ds_skills[skill_name]
            skill_response = chat_skill.invoke(query, dialog_state, instance_code)

        if dialog_state is not None and dialog_state.get('ds_session_id') is not None:
            if skill_name not in list(self.ds_skills.keys()):
                dialog_state['ds_no_response_turns'] = dialog_state.get('ds_no_response_turns', 0) + 1
                if dialog_state.get('ds_no_response_turns', 0) > self.ds_tolerance_turns:
                    dialog_state['ds_session_id'] = None
                    dialog_state['ds_no_response_turns'] = 0
            else:
                dialog_state['ds_no_response_turns'] = 0
        return skill_response, skill_name


if __name__ == '__main__':
    # skill = JokeSkill()
    # print(skill.invoke("明年立秋是什么时间", ""))

    # skill = CalendarSkill()
    # for i in range(10):
    #     print(skill.invoke("明天周几", ""))
    #     print(skill.invoke("今天周几", ""))

    chat_skills = DSSkill()
    response = chat_skills.invoke("天气怎么样", {}, 'geely')
    print(response)

    # print(response)
    # querys = [
    #     '下周一什么天气',
    #     '杭州呢',
    #     '谢谢了',
    #     '今天是几号',
    #     '现在是几点',
    #     '下周二是几号',
    #     '明天是周几',
    #     '好的',
    #     '好的',
    #     '武汉什么天气',
    #     '后天'
    # ]
    # chat_skills = ChatSkills()
    # session_id = None
    # dialog_state = {}
    # for query in querys:
    #     response, name = chat_skills.call(query, query, None, dialog_state)
    #     print(f'{query} >> {response} {dialog_state.get("ds_session_id")} {dialog_state.get("ds_no_response_turns")}')
    # print(get_date("今天"))