import re
from xdpx.utils import register
import importlib
from typing import Dict
from functools import partial
from xdpx.options import Options
from lunar_python import Lunar, Solar, SolarWeek
from datetime import datetime
import sys
from typing import List
from alibabacloud_chatbot20220408.client import Client as Chatbot20220408Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_chatbot20220408 import models as chatbot_20220408_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient
from xdpx.utils import text_is_question


class CalendarSkill(object):
    time_regex = re.compile('现在.*几点')
    day_regex = re.compile('(今天|明天|后天|大后天|昨天|前天|大前天|今日|昨日|明日).*几号')
    day_regex_2 = re.compile('(下周|周|这周|上周|上上周|下下周)([一二三四五六天日]).*几号')
    week_regex = re.compile('(今天|明天|后天|大后天|昨天|前天|大前天|今日|昨日|明日).*(星期几|周几)')
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

    def invoke(self, query, history=None):
        date_now = Solar.fromDate(datetime.now())
        if self.time_regex.findall(query):
            day = date_now
            response = f'现在北京时间是 {day.getYear()}年{day.getMonth()}月{day.getDay()}日{day.getHour()}点{day.getMinute()}分'
            return response, None
        days = self.day_regex.findall(query)
        if days and days[0] in self.day_shift_map:
            day_shift = self.day_shift_map[days[0]]
            day = date_now.next(day_shift)
            response = f'{days[0]}是{day.getYear()}年{day.getMonth()}月{day.getDay()}日'
            return response, None
        days = self.day_regex_2.findall(query)
        if days:
            week_shift = self.week_shift_map[days[0][0]]
            weekday_shift = self.weekday_shift_map[days[0][1]]
            now_weekday = self.weekday_shift_map[date_now.getWeekInChinese()]
            shift = week_shift * 7 + weekday_shift - now_weekday
            day = date_now.next(shift)
            response = f'{days[0][0]}{days[0][1]}是{day.getYear()}年{day.getMonth()}月{day.getDay()}日'
            return response, None

        days = self.week_regex.findall(query)
        if days and days[0][0] in self.day_shift_map:
            day_shift = self.day_shift_map[days[0][0]]
            day = date_now.next(day_shift)
            response = f'{days[0][0]}是周{day.getWeekInChinese()}'
            return response, None
        return None, None


class DSSkill(object):
    def __init__(self):
        config = open_api_models.Config(
            access_key_id='AK',
            access_key_secret='AKS'
        )
        # 访问的域名
        config.endpoint = 'ENDPOINT'
        self.client = Chatbot20220408Client(config)
        self.instance_id = 'INSTANCE_ID'
        self.regex_patterns = [
            re.compile('查.*天气'),
            re.compile('(天气|气温|下雨).*(？|\?|吗|什么|怎么|怎样|咋|啥|如何|为什么|哪|几|谁|多少|多大|多高|是不是|有没有|是否|多久|可不可以|能不能|行不行)'),
            re.compile('(？|\?|吗|什么|怎么|怎样|咋|啥|如何|为什么|哪|几|谁|多少|多大|多高|是不是|有没有|是否|会不会|多久|可不可以|能不能|行不行).*(天气|气温|下雨)'),
            re.compile('带伞吗|下雨吗')

        ]

    def invoke(self, query, history=None):
        should_invoke = False
        session_id = None
        for p in self.regex_patterns:
            if p.findall(query):
                should_invoke = True
                break
        if history is not None:
            for t in history[-5:]:
                if t.get('type') == 'ds':
                    session_id = t.get('ds_session_id')
                    should_invoke = True
        if not should_invoke:
            return None, None

        chat_request = chatbot_20220408_models.ChatRequest(
            instance_id=self.instance_id,
            utterance=query,
            session_id=session_id

        )
        runtime = util_models.RuntimeOptions()
        try:
            response = self.client.chat_with_options(chat_request, runtime)
            session_id = response.body.session_id
            message = response.body.messages[0]
            if message.answer_type == 'Text':
                answer_source = message.text.answer_source
                if 'BotFramework' == answer_source:
                    response = message.text.content
                    return response, session_id

        except Exception as error:
            UtilClient.assert_as_string(error.message)
        return None, session_id


chat_skills = {
    'calendar': CalendarSkill(),
    'ds': DSSkill()
}


def call_chat_skills(query, history):
    for name, chat_skill in chat_skills.items():
        response, session_id = chat_skill.invoke(query, history)
        if response:
            return response, session_id
    return None, None


if __name__ == '__main__':
    querys = [
        '今天是几号',
        '现在是几点',
        '下周二是几号',
        '明天是周几',
        '今天什么天气',
        '杭州呢',
        '谢谢了'
    ]
    session_id = None
    history = []
    for query in querys:
        response, session_id = call_chat_skills(query, history)
        if session_id is not None:
            history.append({
                'type': 'ds',
                'ds_session_id': session_id
            })
        else:
            history.append({})
        print(f'{query} >> {response} {session_id}')