from typing import Callable, Iterable

from ansi.color import fg

SEPARATOR = ' '
CLEAR_SEPARATOR = '|'
TO = '->'


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
    return sep_str(list1, sep) + ' {} '.format(to) + sep_str(list2, sep)