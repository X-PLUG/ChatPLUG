#!/user/bin/env python
# coding=utf-8
'''
@project : XDPX
@author  : fucheng
#@file   : es_search.py
#@ide    : PyCharm
#@time   : 2022-05-20 15:04:41
'''
from elasticsearch import Elasticsearch

'''
初始化ES 检索类需要三个参数，host，port，index
搜索方法的入参为query和size
出参为：
[{
	'keyword_text': '', #百科百科的词条，该词条包括词条的描述+词条
	'content_type': 'content_text', #分为o_text｜spos_list｜content_text｜content_list｜abstract_text五种类型，其中o_text表示召回的content_text为三元组中的object，spos_list表示召回的content_text是一个三元组列表[[subject,predicate,object]],content_text表示召回的是一个段落，content_list表示召回的是一个短路列表，abstract_text表示召回的是词条的摘要描述；
	'content_text': '', # content_text的返回类型由content_type决定
	'content_parent_text': '', # 表示content_text的parent text
	'score': 0 # 召回分数
}]

'''


class ES_Search:
    def __init__(self, index):
        self.index = index
        self.es = Elasticsearch(
            ['es-cn-tl32oudq1000zalhv.public.elasticsearch.aliyuncs.com'],
            http_auth=('elastic', 'virtualHum123'),
            port=9200,
            use_ssl=False
        )

    def search_by_query(self, query, size):
        es_query = {
            'size':
                size,
            "query":
                {"bool":
                    {
                        "should":
                            [
                                {
                                    "match": {"keyword_text": query}
                                },
                                {
                                    "match": {"content_text": query},
                                    "match": {"content_type": "content_text"}
                                }
                            ]
                    }
                }
        }
        query = self.es.search(index=self.index, body=es_query)
        hits = query['hits']['hits']
        results = []
        for hit in hits:
            _source = hit['_source']
            results.append({'keyword_text': _source['keyword_text'], 'content_type': _source['content_type'],
                            'content_text': _source['content_text'], 'content_parent_text': _source['parent_text'],
                            'score': hit['_score']})
        return results


if __name__ == '__main__':
    index = 'baidu_baike_v1'
    size = 5
    es_client = ES_Search(index)
    while True:
        query = input()
        result = es_client.search_by_query(query, size)
        print(result)