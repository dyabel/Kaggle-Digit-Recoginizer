# -*- coding: utf-8 -*-
# @Time    : 2020/12/1 10:23
# @Author  : duyu
# @Email   : abelazady@foxmail.com
# @File    : logger.py
# @Software: PyCharm
import sys
import os


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
