import traceback
from typing import List, Optional

import functools
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

    def __str__(self):  
        return f"{self.sc_name}: {self.snippet}"


class OpenWeb(object):
 
    def __init__(self, search_engine):
        self.cache = {}
        self.search_engine = search_engine

    def search(self, query) -> (List[Snippet], bool):
        """
        Args:
            query: search_query

        Returns:
            list of snippets

        """
        if self.cache and query in self.cache:
            return self.cache[query], False
        snippets, is_special_card = self.search_engine.search(query)
        self.cache[query] = snippets
        return snippets, is_special_card


class BingSearch(object):

    def __init__(self):

        def _request_call(search_term="Microsoft Bing Search Services", search_url="https://api.bing.microsoft.com/v7.0/search"):   
            import os
            import requests
            # bing subscription key
            subscription_key = os.environ.get("BING_SEARCH_API", None)
            assert subscription_key

            headers = {"Ocp-Apim-Subscription-Key": subscription_key}
            params = {"q": search_term, "textDecorations": False, "textFormat": "Raw", "mkl":"zh-CN", "setLang": "zh-hans", "count": 10}
            # "responseFilter": "News"}
            response = requests.get(search_url, headers=headers, params=params)
            response.raise_for_status()
            search_results = response.json()
            # print(search_results)
            # print(json.dumps(search_results, indent=2))
            return search_results


        request_web_call = _request_call
        request_news_call = functools.partial(_request_call, search_url="https://api.bing.microsoft.com/v7.0/news/search")
        self.request_calls = [request_web_call, request_news_call]

    def search(self, query) -> (List[Snippet], bool):
        """ search query

        Args:
            query: search_query

        Returns:
            list of snippets
        """
        snippets = []
        for request_call in self.request_calls:
            try:
                data = request_call(query)
            except Exception as e:
                print(e)
                print(query)
                data = {}

            # 1. parse webpages
            sc_name = 'webPages'
            webPages = data.get(sc_name, {}).get('value', [])
            for page in webPages:
                snippets.append(Snippet(sc_name=sc_name, snippet=page['snippet'], url=page['url']))

            # 2. parse news
            sc_name = 'news'
            news = data.get(sc_name, {}).get('value', [])
            for page in news:
                snippets.append(Snippet(sc_name=sc_name, snippet=f"{page['name']}\n{page['description']}", url=page['url']))
            
            # 3. parse mathematical expression
            sc_name = 'computation'
            if sc_name in data:
                computation = data.get(sc_name)
                snippets.append(Snippet(sc_name=sc_name, snippet=f"{computation['expression']}\n{computation['value']}", url=computation['id']))
            
            # 4. parse timezone
            sc_name = 'timeZone'
            if sc_name in data:
                timeZone = data.get(sc_name).get('primaryCityTime')
                snippets.append(Snippet(sc_name=sc_name, snippet=f"{timeZone['location']}\n{computation['time']}", url='https://www.bing.com/api/v7/#TimeZone'))

            # 5. translation
            sc_name = 'translations'
            if sc_name in data:
                translations = data.get(sc_name)
                snippets.append(Snippet(sc_name=sc_name, snippet=f"originalText: {translations['originalText']}\ntranslatedText: {translations['translatedText']}", url=translations['id']))
            
            # 6. others
            # We ignore it for the sake of simplicity.

        is_special_card = False
        return snippets, is_special_card


if __name__ == '__main__':
    
    bing = BingSearch()
    search = OpenWeb(bing)

    r = search.search("周杰伦是哪里人")
    print('\n\n'.join([str(s) for s in r[0]]))
    print(len(r[0]))

    while True:
        text = input("Query: ")
        r = search.search(text)
        print('\n\n'.join([str(s) for s in r[0]]))
        print(len(r[0]))

