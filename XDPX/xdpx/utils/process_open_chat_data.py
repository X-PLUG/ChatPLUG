import json
import os, time, random
import math
from collections import Counter
import re
import copy
from icecream import ic as icp
from itertools import chain

MAX_PASSAGE_LENGTH = 200
MAX_HISTORY_UTTERANCE = 2  # concat max 2 history turns

name_templates = ['我叫{}', '我的名字是{}', '姓名是{}', '我是{}']
gender_male = ['我是男的，不是女的;', '我是男的呢', '我是男生', '我是个型男', '我是个帅哥']
gender_female = ['我是女的，不是男的;', '我是女的呢', '我是女生', '我是个美女', '我性别是女']
age_templates = ['我今年{}岁了', '我{}岁了', '我年龄{}']


def parse_durecdial_full_profile(s, keys=None, mask_keys=None):
    user_profile = s['user_profile']
    user_profile_str = ''
    for k, v in user_profile.items():
        if keys is not None and k not in keys:
            continue
        if mask_keys is not None and k in mask_keys:
            continue
        if k == '姓名':
            template = random.choice(name_templates)
            user_profile_str += template.format(v) + ';'
        elif k == '性别':
            if v == '男':
                user_profile_str += random.choice(gender_male) + ';'
            else:
                user_profile_str += random.choice(gender_female) + ';'
        elif k == '居住地':
            user_profile_str += random.choice(['我现在住在{};', '我是{}人;']).format(v)
        elif k == '年龄区间':
            goal = s['goal']
            pos = goal.find('问 User 年龄')

            if pos != -1:
                start_pos = goal[:pos].rfind('-->')
                if start_pos != -1:
                    num = goal[start_pos + 3:pos].strip()
                    target_utterance_id = -1
                    for i, c in enumerate(s['conversation']):
                        if c.startswith(num):
                            target_utterance_id = i + 1
                            break
                    if target_utterance_id != -1:
                        age = re.search('\d+', s['conversation'][target_utterance_id])
                        if age:
                            age = age.group(0)
                            user_profile_str += random.choice(age_templates).format(age)
            else:
                ##
                if v == '18-25':
                    user_profile_str += random.choice(age_templates).format(random.randint(18, 25))
                elif v == '26-35':
                    user_profile_str += random.choice(age_templates).format(random.randint(26, 35))
                elif v == '大于50':
                    user_profile_str += random.choice(age_templates).format(random.randint(50, 80))
                elif v == '小于18':
                    user_profile_str += random.choice(age_templates).format(random.randint(10, 18))
                elif v == '36-50':
                    user_profile_str += random.choice(age_templates).format(random.randint(36, 50))
                elif v == '7-10':
                    user_profile_str += random.choice(age_templates).format(random.randint(7, 10))
                else:  # 小于7
                    user_profile_str += random.choice(age_templates).format(random.randint(1, 7))
            user_profile_str += ';'
        elif k == '职业状态':
            # 工作:3125学生:2738退休:755
            if v == '工作':
                user_profile_str += random.choice(['我已经毕业了，现在在工作', '我工作了']) + ';'
            elif v == '学生':
                user_profile_str += random.choice(['我现在在上学', '我是个学生', '我是个学生,还没毕业工作']) + ';'
            else:
                user_profile_str += random.choice(['我已经退休了', '我退休不工作了']) + ';'
        elif k == '喜欢 的 明星' and v:
            if isinstance(v, str):
                v = [v]
            user_profile_str += random.choice(['我喜欢{}', '我喜欢的明星有{}']).format('、'.join(v)) + ';'
        elif k == '同意 的 美食' and v:
            user_profile_str += random.choice(['我喜欢美食，比如{}', '我喜欢吃{}', '我喜欢吃的美食有{}']).format(v) + ';'
        elif (k == '喜欢 的 音乐' or k == '同意 的 音乐' or k == '接受 的 音乐') and v:
            if isinstance(v, str):
                v = [v]
            if len(v) == 1:
                user_profile_str += random.choice(['我喜欢听 {} 这首歌', '我喜欢听歌，比如{}', '我喜欢的歌有:{}']).format('、'.join(v)) + ';'
            else:
                user_profile_str += random.choice(['我喜欢听以下这些歌,{}等等', '我喜欢听歌，比如{}', '我喜欢的歌有:{}']).format(
                    '、'.join(v)) + ';'

        elif (k == '喜欢 的 电影' or k == '同意 的 电影' or k == '接受 的 电影') and v:
            if isinstance(v, str):
                v = [v]
            if len(v) == 1:
                user_profile_str += random.choice(['我喜欢看{}这部电影', '我喜欢看电影，比如{}等', '我喜欢的电影有:{}']).format(
                    '、'.join(v)) + ';'
            else:
                user_profile_str += random.choice(['我喜欢看以下这些电影,{}等等', '我喜欢看电影比如{}等', '我喜欢的电影:{}']).format(
                    '、'.join(v)) + ';'
        elif k == '没有接受 的 音乐' and v:
            if isinstance(v, str):
                v = [v]
            user_profile_str += random.choice(['我不喜欢{}', '我不喜欢听{}', '我不喜欢的音乐有:{}']).format('、'.join(v)) + ';'
        elif k == '没有接受 的 电影' and v:
            if isinstance(v, str):
                v = [v]
            if len(v) == 1:
                user_profile_str += random.choice(['我不喜欢的电影有:{}', '我不喜欢看{}这个电影']).format('、'.join(v)) + ';'
            else:
                user_profile_str += random.choice(['我不喜欢的电影有:{}', '我不喜欢看{}这些电影']).format('、'.join(v)) + ';'
    return user_profile_str


def parse_durecdial(file):
    '''
    durecdial_train = parse_durecdial(durecdial_dir +'/train.txt')
    '''
    data = []
    for l in open(file):
        s = json.loads(l)
        first_goal = s['goal'].split('-->')[0]
        first_utterance_is_user = 'User 主动' in first_goal
        final_goal = s['goal'].split('-->')[-1]
        final_goal_id = final_goal.strip()[:3]
        final_goal_is_bye = '再见' in final_goal

        playmusic_id = ""
        for each_goal in s['goal'].split('-->'):
            playmusic = "播放 音乐" in each_goal
            if playmusic:
                playmusic_id = each_goal.strip()[:3]

        user_profile = parse_durecdial_full_profile(s)

        knowledge_passages = ['']
        user_name = s['user_profile']['姓名']

        _dic = {}
        for t in s['knowledge']:
            if t[0] == user_name:
                if t[2] not in ('poi'):
                    if t[1] not in _dic:
                        _dic[t[1]] = t[2]
                    else:
                        _dic[t[1]] += '、' + t[2]

        for k, v in _dic.items():
            user_profile += ';我{} {}'.format(k, v)

        for t in s['knowledge']:
            if t[0] != user_name:
                tt = ' '.join([t2.replace(' ', '') for t2 in t]) + '</s>'
                knowledge_passages[-1] += tt
                if len(knowledge_passages[-1]) > MAX_PASSAGE_LENGTH:
                    knowledge_passages.append('')

        if '[1]' not in s['conversation'][0]:
            print('WARNING:{}'.format(s['conversation'][0]))

        clean_conversation = [t[3:].replace(' ', '') if t.startswith('[') else t for t in s['conversation']]

        goals_done = []
        for i in range(0, len(s['conversation']) - 1):
            history = clean_conversation[:i]
            utterance = clean_conversation[i]
            response = clean_conversation[i + 1]

            if s['conversation'][i + 1].strip().startswith('['):
                goal_id = s['conversation'][i + 1].strip()[:3]
                goals_done.append(goal_id)
                if final_goal_is_bye and goal_id == final_goal_id:
                    break  # 去除尾轮再见
                if playmusic_id and goal_id == playmusic_id:
                    break  # 去除播放音乐

            if (i % 2 == 0 and first_utterance_is_user) or (i % 2 == 1 and not first_utterance_is_user):
                moshengren = '问 User 姓名' in s['goal']
                banshuren = '提问' in s['goal']

                if not moshengren:
                    if banshuren:  # 半熟人，bot 会提问部分关于user的信息
                        mask = []  # 提问
                        for g in s['goal'].split('-->'):
                            if '提问' in g:
                                mask.append(g.strip()[:3])
                        if len(set(mask) - set(goals_done)) > 0:  # 没问完
                            mask_keys = []  # 提问基本上围绕某个明星提问['新闻', '歌曲', '主演', '电影']，可以全部mask掉，不影响
                            mask_keys.append('喜欢 的 新闻')
                            mask_keys.append('喜欢 的 明星')
                            mask_keys.append('喜欢 的 电影')
                            mask_keys.append('喜欢 的 明星')
                            utterance_profile = parse_durecdial_full_profile(s, mask_keys=mask_keys)
                            data.append({
                                'history': history,
                                'utterance': utterance,
                                'response': response,
                                'knowledge_passages': knowledge_passages,
                                'utterance_profile': utterance_profile
                            })

                        else:  # 问完了
                            data.append({
                                'history': history,
                                'utterance': utterance,
                                'response': response,
                                'knowledge_passages': knowledge_passages,
                                'utterance_profile': user_profile
                            })

                    if not banshuren:  # bot 不提问任何关于user的信息，可以认为是熟人
                        data.append({
                            'history': history,
                            'utterance': utterance,
                            'response': response,
                            'knowledge_passages': knowledge_passages,
                            'utterance_profile': user_profile
                        })


                else:  # 陌生人沟通， utterance_profile抹掉相关内容
                    remain_keys = []
                    for g in goals_done:
                        if g == '[2]':
                            remain_keys.append('姓名')
                        elif g == '[3]':
                            remain_keys.append('性别')
                        elif g == '[4]':
                            remain_keys.append('年龄')
                        elif g == '[5]':
                            remain_keys = None
                    utterance_profile = parse_durecdial_full_profile(s, keys=remain_keys)
                    data.append({
                        'history': history,
                        'utterance': utterance,
                        'response': response,
                        'knowledge_passages': knowledge_passages,
                        'utterance_profile': utterance_profile
                    })
        #             else: #
        #                 response_profile =user_profile
        #                 data.append({
        #                     'history':history,
        #                     'utterance':utterance,
        #                     'response':response,
        #                     'knowledge_passages':knowledge_passages,
        #                     'response_profile':response_profile
        #                 })
        moshengren = '问 User 姓名' in s['goal']
        if moshengren:
            conversation = s['conversation']
            for i in range(len(conversation) - 1):
                c = conversation[i]
                if c[:3] in ('[2]', '[3]', '[4]', '[5]'):
                    u = c[3:].split('，')[-1]
                    if '你' in u:
                        replace_map = {'我': '你', '你': '我'}
                        utterance = ''.join([replace_map[w] if w in replace_map else w for w in u])
                        response = ''.join([replace_map[w] if w in replace_map else w for w in conversation[i + 1]])
                        #                     print('\t{}:{}'.format(utterance,response))
                        data.append({
                            'history': [],
                            'utterance': utterance,
                            'response': response,
                            'utterance_profile': user_profile
                        })

    return data


def parse_dulemon_self(file):
    err = 0
    data = []
    for l in open(file):
        s = json.loads(l)
        persona1_1 = [p[3:] if p.startswith('U') else p for p in s['p1_persona']]
        persona1_2 = [p[3:] if p.startswith('U') else p for p in s['p2_persona']]

        full_p1_persona = ';'.join(persona1_1).replace(' ', '')
        full_p2_persona = ';'.join(persona1_2).replace(' ', '')

        clean_conversation = [t[3:].split('\t')[0] for t in s['conversation']]

        for i in range(0, len(clean_conversation) - 3):  # 最后两轮去掉
            history = clean_conversation[:i]
            utterance = clean_conversation[i]
            response = clean_conversation[i + 1]

            if i % 2 == 0:
                response_profile = full_p2_persona
            else:
                response_profile = full_p1_persona

            data.append({
                'history': history,
                'utterance': utterance,
                'response': response,
                'response_profile': response_profile
            })

    return data


def parse_dulemon_both(file):
    data = []
    for l in open(file):
        s = json.loads(l)
        bot_persona = [p[3:] if p.startswith('B') else p for p in s['bot_persona']]
        user_said_persona = [p[3:] if p.startswith('U') else p for p in s['user_said_persona']]
        user_no_said_persona = [p[3:] if p.startswith('U') else p for p in s['user_no_said_persona']]

        clean_conversation = [t[4:].split('\t')[0] for t in s['conversation']]  # prefix: Usr: Bot:

        for i in range(0, len(clean_conversation) - 1, 2):
            history = clean_conversation[:i]
            utterance = clean_conversation[i]
            response = clean_conversation[i + 1]

            utterance_profile = ';'.join(user_said_persona).replace(' ', '')
            response_profile = ';'.join(bot_persona).replace(' ', '')

            data.append({
                'history': history,
                'utterance': utterance,
                'response': response,
                'utterance_profile': utterance_profile,
                'response_profile': response_profile
            })
    return data


def parse_dureader(file):
    data = []
    for l in open(file):
        d = json.loads(l)
        utterance = d['question']
        answers = d['answers']

        knowledge_passages = []
        for doc in d['documents']:
            for para in doc['paragraphs']:
                knowledge_passages.append(para)

        for answer in answers:
            data.append({
                'history': [],
                'utterance': utterance,
                'response': answer,
                'knowledge_passages': knowledge_passages
            })
    return data


def parse_reco(file):
    data = []
    items = json.loads(open(file).read())
    print(len(items))
    for d in items:
        utterance = d['query']
        answer = d['passage']

        knowledge_passages = [d['doc']]

        if len(answer) < 150:
            data.append({
                'history': [],
                'utterance': utterance,
                'response': answer,
                'knowledge_passages': knowledge_passages
            })
    return data


def parse_dureader_robust(file):
    data = []
    items = json.loads(open(file).read())['data'][0]['paragraphs']
    print(len(items))
    for d in items:
        qas = d['qas']
        knowledge_passages = [d['context']]
        for qa in d['qas']:
            utterance = qa['question']
            answers = qa['answers']
            if answers:
                answer = answers[0]['text']
                if len(answer) < 150:
                    data.append({
                        'history': [],
                        'utterance': utterance,
                        'response': answer,
                        'knowledge_passages': knowledge_passages
                    })
    return data


def parse_dureader_yesno(file):
    data = []
    for l in open(file):
        d = json.loads(l)
        utterance = d['question']
        answer = d['answer']

        knowledge_passages = []
        for doc in d['documents']:
            knowledge_passages.append(''.join(doc['paragraphs']))
        data.append({
            'history': [],
            'utterance': utterance,
            'response': answer,
            'knowledge_passages': knowledge_passages
        })
    return data


def parse_duconv(file):
    data = []
    for l in open(file):
        s = json.loads(l)

        knowledge_passages = ['']
        for t in s['knowledge']:
            tt = ' '.join([t2.replace(' ', '') for t2 in t]) + '</s>'
            knowledge_passages[-1] += tt
            if len(knowledge_passages[-1]) > MAX_PASSAGE_LENGTH:
                knowledge_passages.append('')

        conversation = s['conversation']
        for i in range(1, len(conversation) - 1, 2):  # first utterance is bot
            history = conversation[:i]
            utterance = conversation[i]
            response = conversation[i + 1]

            data.append({
                'history': history,
                'utterance': utterance,
                'response': response,
                'knowledge_passages': knowledge_passages
            })
    return data


def parse_kdconv_kb(file):
    kb = json.loads(open(file).read())
    kb_c = {}
    for name, triples in kb.items():
        dic = {}
        for _, attr, value in sorted(triples):
            if attr not in dic:
                dic[attr] = value
            else:
                dic[attr] += '、' + value
        kb_c[name] = ['{} {} {}'.format(name, attr, value) for attr, value in dic.items()]
    return kb_c


def parse_kdconv(file, kb):
    dialogs = json.loads(open(file).read())
    data = []
    for d in dialogs:
        messages = d['messages']
        for i in range(len(messages) - 1):
            history = [m['message'] for m in messages[:i]]
            utterance = messages[i]['message']
            response = messages[i + 1]['message']
            knowledge_passages = []
            if 'attrs' in messages[i + 1]:
                for attr in messages[i + 1]['attrs']:
                    knowledge_passages.extend(kb[attr['name']])

            data.append({
                'history': history,
                'utterance': utterance,
                'response': response,
                'knowledge_passages': knowledge_passages
            })
    return data


def parse_nconv(file, document_json, dialog_release_json):
    documents = document_json
    dialog_list = dialog_release_json

    dialog_dic = {}
    for d in dialog_list:
        dialog_dic[d['dialog_id']] = d

    ids = [l.strip() for l in open(file)]
    data = []
    for _id in ids:
        dialog = dialog_dic[_id]
        document_id = dialog['document_id']
        document = documents[document_id]
        document_str = '{}\t{}\t{}'.format(document['topic'], document['title'], document['content'])
        dialog_content = dialog['content']
        for i in range(2, len(dialog['content']) - 3):  # 排除首轮的打招呼和尾轮的再见
            history = [m for m in dialog['content'][:i]]
            utterance = dialog['content'][i]
            response = dialog['content'][i + 1]
            knowledge_passages = [document_str]

            data.append({
                'history': history,
                'utterance': utterance,
                'response': response,
                'knowledge_passages': knowledge_passages
            })
    return data


def parse_cconv(file):
    items = json.loads(open(file).read())['data']
    scenarios = [t['scenario'] for t in items]
    types = [t['type'] for t in items]

    print(Counter(scenarios).most_common(100))
    print(Counter(types).most_common(100))
    data = []
    for d in items:
        conversation = d['content']
        for i in range(2, len(conversation) - 1):
            history = conversation[:i]
            utterance = conversation[i]
            response = conversation[i + 1]
            data.append({
                'history': history,
                'utterance': utterance,
                'response': response,
                'utterance_profile': '',
                'response_profile': '',
                'knowledge_passages': []
            })
    return data


def parse_dusinc(file):
    data = []
    for l in open(file):
        s = json.loads(l)
        conversation = s['conversation']
        for i in range(len(conversation) - 1):
            history = [t['utterance'].replace(' ', '') for t in conversation[:i]]
            utterance = conversation[i]['utterance'].replace(' ', '')
            response = conversation[i + 1]['utterance'].replace(' ', '')
            knowledge_passages = conversation[i + 1].get('use_knowledge', '').replace(' ', '')
            if knowledge_passages:
                knowledge_passages = [knowledge_passages]
            else:
                knowledge_passages = []
            data.append({
                'history': history,
                'utterance': utterance,
                'response': response,
                'knowledge_passages': knowledge_passages
            })
    return data


def parse_risawoz(file):
    data = []
    dialogs = json.loads(open(file).read())
    for d in dialogs:

        conversation = d['dialogue']
        for i in range(len(conversation)):
            history = []
            for j in range(i):
                history.append(conversation[j]['user_utterance'])
                history.append(conversation[j]['system_utterance'])

            utterance = conversation[i]['user_utterance']
            response = conversation[i]['system_utterance']

            db_results = conversation[i]['db_results']
            knowledge_passages = []
            if db_results:
                for it in db_results[1:]:
                    try:
                        passage = it.replace('{', '').replace('}', '').replace("\'", "") + '</s>'
                        knowledge_passages.append(passage)
                    except Exception as e:
                        print(it)
                        raise Exception(e)
                        pass
            data.append({
                'history': history,
                'utterance': utterance,
                'response': response,
                'knowledge_passages': knowledge_passages
            })
    return data


def parse_crosswoz_db(crosswoz_dir):
    db = {}
    for domain in ('attraction', 'hotel', 'restaurant'):
        db[domain] = {}
        file_content = open(os.path.join(crosswoz_dir, 'database', f'{domain}_db.json')).read()
        for d in json.loads(file_content):
            name = d[0]
            data = d[1]
            passage = ''
            for k, v in data.items():
                if k != '领域':
                    if isinstance(v, list):
                        v = '、'.join(v)
                    if v:
                        if k == '名称':
                            passage += v
                        else:
                            passage += f'{k}:{v}， '
            db[domain][name] = passage
    return db


def parse_crosswoz(file, db):
    data = []
    dialogs = json.loads(open(file).read())
    for _, d in dialogs.items():
        conversation = d['messages']
        for i in range(len(conversation) - 1):
            history = [t['content'] for t in conversation[:i]]
            utterance = conversation[i]['content']
            response = conversation[i + 1]['content']
            if 'sys_state' in conversation[i + 1]:
                knowledge_passages = []
                for domain, domain_key in zip(['景点', '酒店', '餐馆'], ['attraction', 'hotel', 'restaurant']):
                    selectedResults = conversation[i + 1]['sys_state'].get(domain, {}).get('selectedResults', [])
                    if selectedResults:
                        passage = db.get(domain_key, {}).get(selectedResults[0])
                        if passage:
                            knowledge_passages.append(passage)
                            data.append({
                                'history': history,
                                'utterance': utterance,
                                'response': response,
                                'knowledge_passages': knowledge_passages
                            })
    return data


def parse_ape(file):
    data = []
    for l in open(file):
        d = json.loads(l)
        utterance = d['original_text']
        response = '<equation>' + d['equation']
        data.append({
            'history': [],
            'utterance': utterance,
            'response': response,
            'knowledge_passages': []
        })
    return data


def parse_c3(file):
    data = []
    items = json.loads(open(file).read())
    for d in items:
        conversation = d[0]
        history = [t[2:] for t in conversation]
        last_role = conversation[-1].strip()[0]
        utterance = d[1][0]['question']
        response = d[1][0]['answer']
        if '男的' in utterance:
            if last_role == '男':
                utterance = utterance.replace('男的', '你').replace('女的', '我')
                response = response.replace('他', '我')
            else:
                utterance = utterance.replace('男的', '我').replace('女的', '你')
                response = response.replace('他', '你')

        elif '女的' in utterance:
            if last_role == '男':
                utterance = utterance.replace('男的', '你').replace('女的', '我')
                response = response.replace('他', '你').replace('她', '你')
            else:
                utterance = utterance.replace('男的', '我').replace('女的', '你')
                response = response.replace('他', '我').replace('她', '我')
        elif '他们' in utterance:
            utterance = utterance.replace('他们', '我们')
            response = response.replace('他', '我')
        else:
            continue

        if '不正确' in utterance:
            continue
        if utterance.startswith('下面') or utterance.startswith('下列'):
            continue
        if utterance.startswith('关于你'):
            utterance = '你怎么了'
        if utterance.startswith('关于我'):
            utterance = '我怎么了'

        utterance = utterance.replace('从对话中可以知道', '').replace('从对话可知，', '')

        data.append({
            'history': history,
            'utterance': utterance,
            'response': response,
            'knowledge_passages': []
        })
    return data


def parse_kvpi(file):
    data = []
    for l in open(file).readlines()[1:]:
        ts = l.strip().split("\t")
        utterance = ts[1]
        response = ts[2]
        profile = json.loads(ts[3].replace("'", '"'))
        gender = '男' if profile['gender'] == 'male' else '女'
        constellation = profile['constellation']
        loc = profile['loc']
        response_profile = random.choice(['我是{}的; 我来自{}; 我的星座是{}', '性别:{}; 我住在{}; 星座:{}', '我是{}生;我是{}人;我是{}']).format(
            gender, loc, constellation)

        consistency_Label = ts[5]

        if consistency_Label in ('1'):
            data.append({
                'history': [],
                'utterance': utterance,
                'response': response,
                'response_profile': response_profile
            })

    return data


def parse_gender_chat(file):
    data = []
    for l in open(file):
        ts = l.strip().split('|')
        if len(ts) == 3 and ts[-1] in ('1', '2'):
            if ts[-1] == '1':
                profile = random.choice(gender_male)
            elif ts[-1] == '2':
                profile = random.choice(gender_female)
            data.append({
                'history': [],
                'utterance': ts[0],
                'response': ts[1],
                'response_profile': profile
            })
    return data


def parse_lccc(file):
    data = []
    for l in open(file):
        conversation = json.loads(l)
        if len(conversation) >= 8:
            for i in range(2, len(conversation) - 1):
                history = conversation[:i]
                utterance = conversation[i].replace(' ', '')
                response = conversation[i + 1].replace(' ', '')
                if len(response) > 10:
                    data.append({
                        'history': history,
                        'utterance': utterance,
                        'response': response
                    })
    return data


def extract_persona(line):
    line_split = re.split(r'[。！；？，,;.!\?]', line.strip())
    ss = [t for t in line_split if t.startswith('我') and len(t) > 2]
    return ss


SINGLE_QA_DATA_NAMES = ('dureader', 'dureader_robust', 'dureader_yesno', 'reco', 'ape')


def create_sample(d, search_results_dict, random_profile_func=None):
    source = d.get('source', 'unknown')
    dd = {}
    dd['source'] = source

    response = d['response'] + '</s>'
    dd['response'] = response.replace(' ', '')

    history = d.get('new_history', d.get('history', []))
    utterance = d['utterance'].replace(' ', '')

    concat_context_history = history[-MAX_HISTORY_UTTERANCE:]
    if concat_context_history:
        context = '</s>'.join(concat_context_history) + '</s>' + d['utterance'] + '</s>'
    else:
        context = d['utterance'] + '</s>'
    dd['context'] = context.replace(' ', '')

    dd['passages'] = ''
    concat_context_history2 = history[:-MAX_HISTORY_UTTERANCE]
    if concat_context_history2:
        concat_context_history2 = '</s>'.join(concat_context_history2)[-MAX_PASSAGE_LENGTH:].replace(' ', '')
        dd['passages'] += 'history: ' + concat_context_history2 + ';;;'

    knowledge_passages = d.get('knowledge_passages', [])
    if utterance in search_results_dict:
        knowledge_passages.extend(search_results_dict[utterance])

    for k in knowledge_passages:
        if 'reco' in source or 'dureader' in source:
            ts = k.split('<splitter>')
            if len(ts) == 2:
                k = ts[1]
        chunks = math.ceil(len(k) / float(MAX_PASSAGE_LENGTH))
        for chunk in range(chunks):
            window_size = 10
            start = max(chunk * MAX_PASSAGE_LENGTH - window_size, 0)
            end = chunk * MAX_PASSAGE_LENGTH + MAX_PASSAGE_LENGTH
            dd['passages'] += 'knowledge: ' + k[start: end] + ';;;'

    history = d.get('history', [])
    dynamic_bot_profile = []
    dynamic_user_profile = []
    for k in range(len(history)):
        idx = len(history) - 1 - k
        if k % 2 == 0:  # response
            dynamic_bot_profile.extend(extract_persona(history[idx]))
        else:  # utterance
            dynamic_user_profile.extend(extract_persona(history[idx]))

    dynamic_user_profile.extend(extract_persona(utterance))
    dynamic_bot_profile.extend(extract_persona(response))

    user_profile = d.get('utterance_profile', '')
    bot_profile = d.get('response_profile', '')

    if dynamic_user_profile:
        user_profile += ';' + ';'.join(dynamic_user_profile).replace(' ', '')
    if dynamic_bot_profile:
        bot_profile += ';' + ';'.join(dynamic_bot_profile).replace(' ', '')

    if random_profile_func is not None:
        if not user_profile.strip(";"):
            user_profile = random_profile_func()
        if not bot_profile.strip(";"):
            bot_profile = random_profile_func()

    if user_profile:
        user_profile = user_profile.replace('我', '你')
        dd['passages'] += 'user_profile: ' + user_profile.strip(';') + ';;;'
    if bot_profile:
        dd['passages'] += 'bot_profile: ' + bot_profile.strip(';') + ';;;'
    return dd
