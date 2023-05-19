
# Retrieval

In this section, we will discuss how to enhance ChatPLUG with a search engine or local knowledge base, using the Bing search engine as an example. 


## BING_SEARCH_API

To use Bing search, we need to obtain a subscription key by following the official documentation provided in the links for an overview and to apply for the API. Once we have the key, we can export it using the provided command.

- [overview](https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview)
- [apply api](https://portal.azure.com/#home)

```bash
export BING_SEARCH_API=<subscription_key>
```

## OpenWeb 

The `OpenWeb` class provides a wrapper for the Bing search engine to return a list of text snippets.

```python
class OpenWeb(object):
 
    def __init__(self, search_engine=None):
        if search_engine is None:
            search_engine = BingSearch()
        self.search_engine = search_engine
    
    @lru_cache
    def search(self, query) -> (List[Snippet], bool):
        """
        search with cache.

        Args:
            query: search_query

        Returns:
            snippets: list of snippets
            is_special_card: True/False
        """
        return self.search_engine.search(query)
```


## Learn2Search

To improve performance, we can implement a `query classifier` and `query rewriter` model to determine whether a question needs to be searched and rewrite it as a suitable search query. 

For simplicity, we will use `text_is_question` as the `query classifier` and `the last utterance` as the `search query`. For better performance, we may need to build our own `query classifier` and `query rewriter`.

```python
class BaseLearn2Search(object):
    def __init__(self):
        print(f'| skip query_classifier.')    
        self.query_classifier = None

    def need_search(self, query: str) -> Tuple[bool, str]:
        return text_is_question(query) and not is_persona_question(query), CHITCHAT_QUERY
    
    def get_search_query(self, query: str, history: List[HistoryItem]):
        # only use the last query
        return query
```

## Config

To configure ChatPLUG to use Bing search, we need to enable the use of the `OpenWeb` class in the `chatplug_3.7B_sftv2.6.0_instruction.hjson` file. We also need to specify the directory and provider for the utterance rewriter and provide the path for the learn2search query classifier.

```hjson
openweb_use: true

# rewrite
utterance_rewriter_save_dir: ""
utterance_rewriter_is_onnx: false
utterance_rewriter_quantized: false
utterance_rewriter_provider: cuda

# learn2search
learn2search_query_classifier_path: ""
```

Overall, these steps will allow us to enhance ChatPLUG with a search engine or local knowledge base, providing users with more accurate and helpful responses.