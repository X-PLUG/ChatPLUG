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
    
    def __init__(self):
        pass    

    def search(self, query) -> List[Tuple[str, str, str]]:
        '''
        TODO: implement it using opensource KG

        Args:
            query: subject entity
        Returns:
            list of (subject, predicate, object) tuples
        '''
        return []
