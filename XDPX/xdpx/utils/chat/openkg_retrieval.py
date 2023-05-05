from typing import List, Tuple

import requests
import json
from dataclasses import dataclass


@dataclass
class Triple(object):
    subject: str
    predicate: str
    object: str


class OpenKG:
    ##云上服务器只能采用http + cookie的方式访问弹内ES
    def __init__(self):
        self.host = "es-cn-751b36x4z601u5403.kibana.elasticsearch.aliyuncs.com:5601"
        self.index_name = "deepqa_ownthink_kg_whole"
        self.url = 'https://{}/api/console/proxy?path=%2F{}%2F_search&method=POST'.format(self.host, self.index_name)
        self.cookie = 'cna=4d3AGTDwzWICASp4Smc4yzO7; sid=Fe26.2**ef7c651c27d4f13934c0214c3758145ded7e515e4a8141c03a6d8dea9c4b3d56*82d3kUZJSMgO-DF77bSjTw*r_cfa_NnGejyztjgzqhPa6AXR4wUGrijkRze_GdDzWOB' \
                      'scfUTXLWXgPdK1_ThfrrYj_WsewveDWKQDsWc1HLS1gh2OMFAkB1mPKwM6ya53azfxKPMzMTDI9nYIS9WHNDfoQughF7HjNdl2zUr5nlOeghCiz56ApRbEXt_4lUxf0**3258d964a0af5c76b39ab65fd' \
                      '92d9c94458ee131215c432c9770b65382f1c9c7*LBHEbjznhwc1Sz0-IgXuaoD8hoVBy6ry3Qx89W4j0nc'
        self.header = {
            'Referer': "https://es-cn-751b36x4z601u5403.kibana.elasticsearch.aliyuncs.com:5601/app/kibana",
            "kbn-version": "6.7.0",
            "Content-Type": "application/json"
        }

    def search(self, query) -> List[Tuple[str, str, str]]:
        '''

        Args:
            query: subject entity
        Returns:
            list of (subject, predicate, object) tuples
        '''
        param = {"size": 1000,
                 "query": {
                     "term": {"subject": query}
                 }}
        try:
            res = requests.post(self.url, data=json.dumps(param), cookies={"cookie": self.cookie}, headers=self.header)
            res_data = json.loads(res.text)
        except Exception as e:
            print(e)
            return []
        hit_num = res_data["hits"]["total"]
        if hit_num == 0:
            return []
        else:
            results = []
            for each_data in res_data["hits"]["hits"]:
                subject = each_data["_source"]["subject"]
                predicate = each_data["_source"]["predicate"]
                object = each_data["_source"]["object"]
                results.append((subject, predicate, object))
        return results
