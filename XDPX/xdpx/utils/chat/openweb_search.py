import traceback
from typing import List, Optional

from googlesearch import quote_plus
import requests
import json
from dataclasses import dataclass, field


@dataclass
class Snippet:
    snippet: str
    sc_name: Optional[str] = field(default='')
    url: Optional[str] = field(default='')
    score: Optional[float] = field(default=0.0)


class OpenWeb(object):
    '''
        shenma search
    '''
    DEFAULT_SESSION = requests.Session()
    DEFAULT_SESSION.cookies = requests.utils.cookiejar_from_dict(
        {'sm_uuid': '45f5c34c3719d39b4003f636b8f5a72e%7C%7C%7C1653558899'}, cookiejar=None, overwrite=True)

    def __init__(self, is_test=False):
        if is_test:
            self.url_search = "https://test.m.sm.cn/api/s_api?q={}&ft=json&no_html=1&ad=no&osr=damoyuan&sl=23&uid=45f5c34c3719d39b4003f636b8f5a72e%7C%7C%7C1653558899"
        else:
            # 采用Quark搜索
            self.url_search = "https://agg.sm.cn/api/s_api?q={}&ft=json&no_html=1&ad=no&osr=damoyuan&sl=7&uid=45f5c34c3719d39b4003f636b8f5a72e%7C%7C%7C1653558899&from=kkframenew&belong=quark"
        self.cache = {}

    def wrap_snippet(self, sc_name, snippet, url=None):
        if snippet:
            if not isinstance(snippet, str):
                print(f'warning: {snippet} not string , ignored')
                return None
            snippet = snippet.strip().replace('<em>', '').replace('</em>', '').replace('...', '').replace('</li>', '').replace('<li>', '')
            if snippet:
                return Snippet(sc_name=sc_name, snippet=snippet, url=url)
        return None

    def search(self, query) -> (List[Snippet], bool):
        '''

        Args:
            query: search_query

        Returns:
            list of snippets

        '''
        # 暂时不用cache，之后会用
        # if self.cache and query in self.cache:
        #     return self.cache[query], False

        # is_recent_ques = False
        # for each in ["最近", "最新", "近期", "2022", "今年", "这个月", "今天"]:
        #     if each in query:
        #         is_recent_ques = True

        for each in ["最近", "最新", "近期", "今年", "这个月", "今天", "世界杯"]:
            if each in query:
                query = "2022 " + query

        query = quote_plus(query)
        url = self.url_search.format(query)
        # text = get_page(url)
        text = self.DEFAULT_SESSION.get(url).text
        try:
            data = json.loads(text)
        except Exception as e:
            print(f'parse json {url} error')
            print(e)
            print(text)
            data = {}

        items = data.get('items', {}).get('item', [])
        snippets = []

        for item in items:
            title = item.get('title', '')
            if title and '视频' in title:
                continue
            sc_name = item.get('sc_name', '')
            if sc_name in ('news_natural', 'structure_web_bbs', 'text_recommend', 'short_video',
                           'structure_short_video', 'xiami_lyric_song', 'kg_recommend_n', 'structure_doc',
                           'kg_recommend_dim_1',
                           'medical_hq_video', 'kg_recommend_n', 'doc_sc_0', 'doc_sc_1', 'doc_sc_3'
                           ):
                continue
            # desc = item.get('desc', '')
            # snippet = f'{title} {desc}'.strip()
            # snippets.append(wrap_snippet()(sc_name, snippet))
            url = item.get('url', '')
            # if not is_recent_ques:
            #     snippets.append(self.wrap_snippet(sc_name, item.get('desc'), url))
            snippets.append(self.wrap_snippet(sc_name, item.get('desc'), url))
            try:
                if sc_name:
                    data = item.get(sc_name)
                    if sc_name == 'weather_new_huake':
                        city = data.get('item', {}).get('city', '')
                        for i in range(1, 2):  # only today
                            dkey = f'day_{i}'
                            it = data.get('item', {}).get(dkey, {})
                            dname = it.get('week_day', '')
                            weather2 = it.get('weather2', '')
                            temp = it.get('temp', '')
                            windPowerLevel = it.get('windPowerLevel', '')
                            windDirection = it.get('windDirection', '')
                            snippet = f'{dname}{city}天气{weather2} 气温{temp} {windDirection}{windPowerLevel}'
                            snippets.append(self.wrap_snippet(sc_name, snippet, url))
                        break
                    elif sc_name == 'weather_moji':
                        city = data.get('wisdomData', {}).get('city', '')
                        break
                    elif sc_name == 'finance_stock_new':
                        name = data.get('Name', '')
                        moduleData = data.get('moduleData', {})
                        gc = {d['label']: d['value'] for d in moduleData.get('grid_container', {})}
                        zuidi = gc.get('最低', '')
                        zuigao = gc.get('最高', '')
                        zuoshou = gc.get('昨收', '')
                        jinkai = gc.get('今开', '')
                        shizhi = gc.get('市值', '')
                        snippet = f'{name}股价, 昨天收盘价{zuoshou}元, 今日开盘价{jinkai}元'
                        snippets.append(self.wrap_snippet(sc_name, snippet, url))
                    elif sc_name == 'wenda_selected':
                        snippet = ""  # 不拼接问答Title
                        # snippet = data.get('item', {}).get('name', '')
                        answer = data.get('item', {}).get('answer', {})
                        if isinstance(answer, str):
                            snippet += answer
                        else:
                            answer = answer.get('item', '')
                            if isinstance(answer, list):
                                snippet += ' '.join(answer)
                            elif isinstance(answer, str):
                                snippet += answer
                        snippets.append(self.wrap_snippet(sc_name, snippet, url))
                    elif sc_name == 'structure_web_info':
                        pass
                        # if is_recent_ques:
                        # try:
                        #     time_stamp1 = item.get('time', '')
                        #     time1 = time.localtime(float(time_stamp1))
                        #     year = time1[0]
                        #     month = time1[1]
                        #     day = time1[2]
                        # except:
                        #     year = None
                        # # if not year:
                        # time2 = data.get('time')
                        # if time2:
                        #     year = time2.split("年")[0]
                        # if year:
                        #     if int(year) >= 2022:
                        #         snippet = self.wrap_snippet(sc_name, data.get('V_MAIN_BODY'), url)
                        #         snippets.append(snippet)
                    elif sc_name == 'structure_web_how':
                        snippet = ''
                        for k in ('SP_HOW_STEP_FIRST', 'SP_HOW_STEP_SECOND', 'SP_HOW_STEP_THIRD', 'SP_HOW_STEP_FOURTH'):
                            snippet += data.get(k, '')
                        snippets.append(self.wrap_snippet(sc_name, snippet, url))
                    elif sc_name == 'yisou_film':
                        film_name = data.get('name', '')
                        brief = data.get('brief', '')
                        directors = data.get('directors', [])
                        actors = data.get('actors', [])
                        snippet = film_name + brief
                        if directors:
                            snippet += '导演是' + '、'.join(directors)
                        if actors:
                            snippet += '主演是' + '、'.join(actors)
                        snippets.append(self.wrap_snippet(sc_name, snippet, url))
                    elif sc_name == 'baike_sc':
                        name = data.get('name', '')
                        abstract = data.get('abstract', '')
                        if abstract:
                            snippets.append(self.wrap_snippet(sc_name, abstract, url))
                        else:
                            text = item.get('moduleData', {}).get('baike_info', '')
                            url = item.get('moduleData', {}).get('baike_url', '')
                            snippets.append(self.wrap_snippet(sc_name, text, url))
                        basic = data.get('basic', [])

                        if basic:
                            kv_info = ''
                            for kv in basic:
                                key = kv.get('key')
                                value = kv.get('value')
                                kv_info += f'{name} {key} {value} </s>'
                            snippets.append(self.wrap_snippet(sc_name, kv_info, url))
                    elif sc_name == 'peoplestarzeus':
                        kg_data = data.get('kg_data', {})
                        name = kg_data.get('name', '')
                        notable_for = kg_data.get('notable_for', '')
                        date_of_birth_with_age = kg_data.get('date_of_birth_with_age', '')
                        rel_person = kg_data.get('rel_person', {}).get('item', [])
                        snippet = f'{name} {notable_for} 出生日期 {date_of_birth_with_age}</s>'
                        for desc_name in rel_person:
                            name2 = desc_name.get('name', '')
                            desc = desc_name.get('desc', '')
                            snippet += f'{name} {desc} {name2}</s>'
                        snippets.append(self.wrap_snippet(sc_name, snippet, url))
                    elif sc_name == 'kk_kg_entity_people':
                        snippet = data.get('sense_name', ' ') + data.get('abstract', ' ')
                        name = data.get('name')
                        baike_kv = data.get('baike_kv', {}).get('item', [])
                        if name and baike_kv:
                            triples = ['{} {} {}'.format(name, d.get('label', ''), d.get('value', '')) for d in
                                       baike_kv]
                            snippet += '</s>'.join(triples)
                        snippets.append(self.wrap_snippet(sc_name, snippet, url))
                    elif sc_name == 'news_uchq':
                        display = data.get('display', {}).get('summary', '')
                        source = data.get('display', {}).get('source', '')
                        if display:
                            snippet = f'{source}消息: {display}'
                            snippets.append(self.wrap_snippet(sc_name, snippet, url))
                        news_node = data.get('news_node', [])
                        for node in news_node[:3]:
                            time = node.get("time", '')
                            if '分钟前' in time or '小时前' in time:
                                snippet = '{} {}'.format(node.get('title'), node.get('summary'))
                                snippets.append(self.wrap_snippet(sc_name, snippet, url))
                    elif sc_name == 'news_top_list':
                        news = data.get('display', {}).get('list_info', {}).get('fields', [])
                        for new in news[:20]:
                            if type(new) == dict:
                                title = new.get('title', '')
                                summary = new.get('news_summary', '')
                                snippet = f'头条新闻: {title};;;新闻摘要: {summary}'
                                snippets.append(self.wrap_snippet(sc_name, snippet, url))

                    elif sc_name == 'covid_19':
                        try:
                            tab_container = data.get('wisdomData', {}).get('tab_container', [])

                            if tab_container and type(tab_container[0]) == dict:
                                data_new = tab_container[0].get('data_new', '[]')
                                data_new = json.loads(data_new)
                                text_new = data.get('wisdomData', {}).get('text_new', '')
                                snippet = f'{text_new},'
                                for d in data_new:
                                    title = d.get('title', '')
                                    val = d.get('val', '')
                                    snippet += f'{title}{val}, '
                                snippets.append(self.wrap_snippet(sc_name, snippet, url))
                                break
                        except:
                            print(f'warning: covid_19 card parse error')
            except Exception as e:
                print(e)
                print(traceback.print_exc())
                print(f'warning parse error')

        snippets = [t for t in snippets if
                    t is not None and t.snippet is not None]
        is_special_card = False

        filtered_snippet = []
        for snippet in snippets:
            if snippet.sc_name in ('weather_new_huake', 'finance_stock_new', 'covid_19'):
                filtered_snippet.append(snippet)
        if filtered_snippet:
            is_special_card = True
            return filtered_snippet, is_special_card
        return snippets, is_special_card


if __name__ == '__main__':
    ""
    search = OpenWeb()
    r = search.search("周杰伦是哪里人")
    print([s.snippet for s in r[0]])
    print(len(r[0]))