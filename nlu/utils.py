from typing import Callable, Iterable

from ansi.color import fg

import logging
import time
import re


SEPARATOR = ' '
CLEAR_SEPARATOR = '|'
TO = '->'
TO_SEP = ' {} '.format(TO)


def bracket(text):
    return '[' + text + ']'


def list_to_str(strlist, sep=' '):
    return sep.join([str(s) for s in strlist])


def colorize_list(arr, begin=None, end=None, color: Callable=fg.blue) -> str:
    if begin is None:
        begin = arr[0]
    if end is None:
        end = arr[-1]
    return list_to_str(arr[:begin]) + ' ' + color(list_to_str(arr[begin:end + 1])) \
           + ' ' + list_to_str(arr[end + 1:])


def id_incrementer():
    id_ = 0
    while True:
        yield id_
        id_ += 1


def overrides(overridden):
    def overrider(method):
        assert (method.__name__ in dir(overridden)), f'method "{method.__name__}" did not override method in class "' \
            f'{overridden.__name__}"'
        return method
    return overrider


def sep_str(list_: Iterable, sep=None):
    if sep is None:
        sep = SEPARATOR
    return sep.join([str(member) for member in list_]) if list_ else ''


def ls_to_ls_str(list1: Iterable, list2: Iterable, sep=None, to=None):
    if to is None:
        to = TO
    if sep is None:
        sep = CLEAR_SEPARATOR
    return sep_str(list1, sep) + TO_SEP + sep_str(list2, sep)


def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s')

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def logtime(logger_name):
    def timeit(method):
        def timed(*args, **kw):
            ts = time.time()
            logging.getLogger(logger_name).info("Running function {}".format(method.__name__))
            result = method(*args, **kw)
            te = time.time()
            logging.getLogger(logger_name).info("function {} ran in {}s".format(method.__name__, round(te - ts, 2)))
            return result
        return timed
    return timeit


def lprint(*args, **kwargs):
    import inspect
    import os
    import sys
    callerFrame = inspect.stack()[1]  # 0 represents this line
    myInfo = inspect.getframeinfo(callerFrame[0])
    myFilename = os.path.basename(myInfo.filename)
    print('{}({}):'.format(myFilename, myInfo.lineno), *args, flush=True, file=sys.stderr, **kwargs)
