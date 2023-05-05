# -*- coding: utf-8 -*- 
"""
@Time : 2022-06-29 16:45 
@Author : zhimiao.chh 
@Desc : 
"""


"""
chat_log.py
Authors: tjf141457 (tjf141457@alibaba-inc.com)
1、git clone http://gitlab.alibaba-inc.com/mit-semantic/common.git
2、pip install -e .
3、把多个log文件放到文件夹并设置root_dir变量
4、运行代码
"""
import os, sys
from common import data_io
from icecream import ic
def self_chat_log_2excel(root_dir="/code/test/tmp",output_file='chat_log.xlsx'):
    columns = []
    data = []
    for file in os.listdir(root_dir):
        path = os.path.join(root_dir, file)
        ic(file)
        columns.append(file)
        lines = open(path).readlines()
        sessions, session = [], []
        for line in lines:
            if line.startswith(">>> START SELF CHAT 6 TURNS <<<"):
                if len(session):
                    sessions.append("\n".join(session))
                session = []
            else:
                session.append(line.strip())
        ic(sessions[0])
        ic(len(sessions))
        data.append(sessions)

    data = list(zip(*data))
    data_io.write(data, columns=columns, file_name=output_file)

if __name__ == '__main__':
    ""
    self_chat_log_2excel()