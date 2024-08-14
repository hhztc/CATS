#! /usr/bin/env python
#encoding:utf-8
import os


def create_dir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir