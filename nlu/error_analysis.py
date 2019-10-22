import collections
from typing import Dict

from nlu.error_structure import MentionTypeError, MergeSplitError, FalseError, SpanError, NERCorrect, NERErrorComposite, \
    ComplicateError


def is_error(ems_pair):
    return isinstance(ems_pair.result, NERErrorComposite)

def is_correct(ems_pair):
    return isinstance(ems_pair.result, NERCorrect)

def is_span_error(ems_pair):
    return is_error(ems_pair) and isinstance(ems_pair.result.span_error, SpanError)

def is_false_error(ems_pair):
    return is_error(ems_pair) and isinstance(ems_pair.result.false_error, FalseError)

def is_ex(ems_pair):
    return is_span_error(ems_pair) and ems_pair.result.span_error.span_type == 'Expansion'

def is_re(ems_pair):
    return is_ex(ems_pair) and ems_pair.result.span_error.direction == 'Right'

def is_le(ems_pair):
    return is_ex(ems_pair) and ems_pair.result.span_error.direction == 'Left'

def is_rle(ems_pair):
    return is_ex(ems_pair) and ems_pair.result.span_error.direction == 'Right Left'

def is_dim(ems_pair):
    return is_span_error(ems_pair) and ems_pair.result.span_error.span_type == 'Diminished'

def is_rd(ems_pair):
    return is_dim(ems_pair) and ems_pair.result.span_error.direction == 'Right'

def is_ld(ems_pair):
    return is_dim(ems_pair) and ems_pair.result.span_error.direction == 'Left'

def is_rld(ems_pair):
    return is_dim(ems_pair) and ems_pair.result.span_error.direction == 'Right Left'

def is_cross(ems_pair):
    return is_span_error(ems_pair) and ems_pair.result.span_error.span_type == 'Crossed'

def is_rc(ems_pair):
    return is_cross(ems_pair) and ems_pair.result.span_error.direction == 'Right'

def is_lc(ems_pair):
    return is_cross(ems_pair) and ems_pair.result.span_error.direction == 'Left'

def is_merge_split(ems_pair):
    return is_error(ems_pair) and isinstance(ems_pair.result.span_error, MergeSplitError)

def is_merge(ems_pair):
    return is_merge_split(ems_pair) and ems_pair.result.span_error.type == 'Spans Merged'

def is_split(ems_pair):
    return is_merge_split(ems_pair) and ems_pair.result.span_error.type == 'Span Split'

def is_type_error(ems_pair):
    return is_error(ems_pair) and isinstance(ems_pair.result.type_error, MentionTypeError)

def filtered_results(parser, is_func):
    span_errors = []
    for doc in parser.docs:
        for sent in doc:
            if sent.ems_pairs:
                for ems_pair in sent.ems_pairs:
                    try:
                        if is_func(ems_pair):
                            span_errors.append(ems_pair.result)
                    except NameError:
                        raise NameError('is_func is not one of the predefined result types')

    return span_errors

def get_all_span_errors(parser):
    _is_types = {'Right Expansion': is_re, 'Left Expansion': is_le, 'Right Left Expansion': is_rle,
                 'Right Diminished': is_rd, 'Left Diminished': is_ld, 'Right Left Diminished': is_rld,
                 'Right Crossed': is_rc, 'Left Crossed': is_lc,
                 'Spans Merged': is_merge, 'Span Split': is_split,
                 'All Span Errors': is_span_error}
    return {name: filtered_results(parser, is_func) for name, is_func in _is_types.items()}

def get_all_false_errors(parser):
    _is_types = {'False Positive': is_re, 'False Negative': is_le, 'All False Errors': is_span_error}
    return {name: filtered_results(parser, is_func) for name, is_func in _is_types.items()}

def print_list_len_in_dict(dict_: Dict):
    for name, list_ in dict_.items():
        print('{}: {}'.format(name, len(list_)))

def get_all_type_errors(parser):
    _is_types = {'All Type Errors': is_type_error}
    return {name: filtered_results(parser, is_func) for name, is_func in _is_types.items()}

def get_all_span_and_type_errors(parser):
    _is_types = {'All Span and Type Errors': lambda x: is_type_error(x) and is_span_error(x)}
    return {name: filtered_results(parser, is_func) for name, is_func in _is_types.items()}

def get_type_errors(parser):
    all_type_errors = get_all_type_errors(parser)

    error_table = {}

    for error in all_type_errors['All Type Errors']:
        type = str(error.type_error)
        if type not in error_table.keys():
            error_table[type] = 1
        else:
            error_table[type] += 1

    for key in sorted(error_table.keys()):
        print(key + ": " + str(error_table[key]))


class NERErrorAnalyzer:
    """
    input: DocumentsWithErrorAnn
    """

    @staticmethod
    def analyze(parser):

        print('-----------------------')

        all_span_errors = get_all_span_errors(parser)
        get_type_errors(parser)
        all_type_errors = get_all_type_errors(parser)
        all_span_and_type_errors = get_all_span_and_type_errors(parser)

        print_list_len_in_dict(all_span_errors)
        print_list_len_in_dict(all_type_errors)
        print_list_len_in_dict(all_span_and_type_errors)

        # get FP/FN
        groups = collections.defaultdict(list)
        false_count = 0
        for doc in parser.docs:
            for sentence in doc:
                if sentence.ems_pairs:
                    for ems_pair in sentence.ems_pairs:
                        if is_false_error(ems_pair):
                            groups['{} - {}'.format(ems_pair.result.false_error.false_type,
                                                    ems_pair.result.false_error.em_type)].append(ems_pair)
                            false_count += 1

        # print total number in each list in a dictionary
        for name, group in groups.items():
            print('{}: {}'.format(name, len(group)))
        print("Total false positive/negative: " + str(false_count))

        # get complicated error
        for doc in parser.docs:
            for sentence in doc:
                if sentence.ems_pairs:
                    for ems_pair in sentence.ems_pairs:
                        if is_error(ems_pair) and isinstance(ems_pair.result.span_error, ComplicateError):
                            print(ems_pair.result.span_error)

        return all_span_errors, all_type_errors, all_span_and_type_errors

