#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Singleton(type):
    # For more info : https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]