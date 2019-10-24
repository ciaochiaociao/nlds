from collections import defaultdict
from typing import Callable, Iterable

from ansi.color import fg

import logging
import time
import re

from pandas import DataFrame

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


def groupby(list_, key):
    groups = defaultdict(list)
    for i in list_:
        groups[key(i)].append(i)

    return groups


def sort_two_level_group(df: DataFrame, indexorder, columnorder):
    return df.reindex(indexorder, columns=columnorder)


def two_level_groupby(list_, key1, key2, count=False, apply_func=False):
    g1 = groupby(list_, key1)
    g2 = {type_: groupby(err, key2) for type_, err in g1.items()}

    if count:
        apply_func = len

    if apply_func:
        g2 = {_: {_: apply_func(v2) for _, v2 in v.items()} for _, v in g2.items()}

    return g2


def add_total_column_row(df: DataFrame):

    df.loc['Total'] = df.sum()
    df['Total'] = df.T.sum()


def frame_text(text, markers=None, style='=', spaces=[0, 1, 0, 1]):
    markers_dict = {
        '=': ['╔', '═', '╗', '║', '║', '╚', '═', '╝'],
        '+-': ['+', '-', '+', '|', '|', '+', '-', '+'],
        '+': ['+'] * 8,
        'x': ['x'] * 8,
        'o': ['o'] * 8
    }
    if not markers or len(markers) != 8 or any([not isinstance(marker, str) for marker in markers]):
        markers = markers_dict[style]

    tl, tm, tr, ml, mr, bl, bm, br = markers
    lines = text.split('\r\n')
    width = max([len(line) for line in lines])
    height = len(lines)
    boxw = width + spaces[1] + spaces[3] + len(tl) + len(tr)
    print(tl + tm * (boxw - len(tl) - len(tr)) + tr)

    for i in range(spaces[0]):
        print(ml + ' ' * (boxw - len(ml) - len(mr)) + mr)
    for y in range(height):
        trailing = boxw - len(lines[y]) - spaces[3] - len(ml) - len(mr)
        print(ml + ' ' * spaces[3] + lines[y] + ' ' * trailing + mr)
    for i in range(spaces[2]):
        print(ml + ' ' * (boxw - len(ml) - len(mr)) + mr)
    print(bl + tm * (boxw - len(tl) - len(tr)) + br)


def fprint(text, *args, **kwargs):
    """
    print framed text
    :param text:
    :param args:
    :param kwargs:
    :return: None
    """
    print(frame_text(text, *args, **kwargs))


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct


def calc_f1(corrects, pred_total, true_total):

    precision = corrects / pred_total
    recall = corrects / true_total
    f1 = 2 * precision * recall / (recall + precision)

    return precision, recall, f1
