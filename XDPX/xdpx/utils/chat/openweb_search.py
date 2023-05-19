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
 
    def __init__(self, is_test=False):
        self.cache = {}

    def search(self, query) -> (List[Snippet], bool):
        """
        TODO: implement this using Bing Search.

        Args:
            query: search_query

        Returns:
            list of snippets

        """
        return [], False

if __name__ == '__main__':
    ""
    search = OpenWeb()
    r = search.search("周杰伦是哪里人")
    print([s.snippet for s in r[0]])
    print(len(r[0]))