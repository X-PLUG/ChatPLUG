'''
base dataobject class
base util functions
'''
from dataclasses import dataclass
from typing import List, Optional
from xdpx.utils.chat import text_is_bye
from dataclasses import field

BAIKEQA_QUERY = 'baike_qa'
CHITCHAT_QUERY = 'chitchat'


@dataclass
class HistoryItem(object):
    utterance: str = ''
    role: str = 'bot'
    rewritten_utterance: Optional[str] = None
    # response类型，model / faq / skill / rule
    type : Optional[str] = None

@dataclass
class ChatConfig(object):
    local_retrieval_use : bool = True

@dataclass
class ChatInput(object):
    query: str = ''
    history: List[HistoryItem] = None
    dialog_state: dict = field(default=dict)
    bot_profile: str = field(default='')
    user_profile: str = field(default='')
    chat_config: ChatConfig = ChatConfig()

    def history_as_string_list(self):
        if self.history:
            return [h.utterance for h in self.history]
        return []

    def history_as_string_list2(self):
        if self.history:
            return [h.rewritten_utterance or h.utterance for h in self.history]
        return []

    def remove_bye_in_history(self):
        if self.history:
            for h in self.history:
                if text_is_bye(h.utterance):
                    h.utterance = h.rewritten_utterance = '#'

    def history_filter_as_user_said(self):
        if self.history:
            return [h.rewritten_utterance or h.utterance for h in self.history if h.role != 'bot']
        return []

    def history_filter_as_bot_said(self):
        if self.history:
            return [h.rewritten_utterance or h.utterance for h in self.history if h.role == 'bot']
        return []

    def pretty_print(self):
        pass


@dataclass
class Evidence(object):
    obj: object = None
    score: float = 0.0
    source: str = ''


@dataclass
class ChatOutput(object):
    response: str = ''
    grounding_evidence: Optional[Evidence] = None
    response_type: Optional[str] = ''
    dialog_state: dict = field(default=dict)
    debug_info: dict = field(default=dict)
