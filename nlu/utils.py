from typing import Callable

from ansi.color import fg


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

