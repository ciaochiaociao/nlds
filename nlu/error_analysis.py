from typing import Dict, List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
import numpy as np

from nlu.error_structure import MentionTypeError, MergeSplitError, FalseError, SimpleSpanError, NERCorrect, NERErrorComposite, \
    ComplicatedError
from nlu.ext_utils.confusion_matrix_pretty_print import pretty_plot_confusion_matrix
from nlu.parser import ConllParser
from nlu.utils import groupby, two_level_groupby, sort_two_level_group, add_total_column_row, fprint, make_autopct

cmap = sns.cubehelix_palette(as_cmap=True, light=.9)


def is_error(ems_pair):
    return isinstance(ems_pair.result, NERErrorComposite)

def is_correct(ems_pair):
    return isinstance(ems_pair.result, NERCorrect)

def is_span_error(ems_pair):
    return is_error(ems_pair) and isinstance(ems_pair.result.span_error, SimpleSpanError)

def is_only_span_error(ems_pair):
    return is_span_error(ems_pair) and not is_type_error(ems_pair) and not is_complicate_error(ems_pair)

def is_false_error(ems_pair):
    return is_error(ems_pair) and isinstance(ems_pair.result.false_error, FalseError)

def is_only_type_error(ems_pair):
    return is_type_error(ems_pair) and not is_span_error(ems_pair) and not is_complicate_error(ems_pair)

def is_fn(ems_pair):
    return is_false_error(ems_pair) and ems_pair.result.false_error.false_type == 'False Negative'

def is_fp(ems_pair):
    return is_false_error(ems_pair) and ems_pair.result.false_error.false_type == 'False Positive'

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
    return is_span_error(ems_pair) and 'Crossed' in ems_pair.result.span_error.type

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

def is_complicate_error(ems_pair):
    return is_error(ems_pair) and isinstance(ems_pair.result.span_error, ComplicatedError)


def filtered_results(parser, is_funcs, boolean=all) -> List[NERErrorComposite]:
    """
    :param parser: `ConllParser` class
    :param is_funcs: filter functions that return boolean value
    :param boolean: `any` or `all`
    :return: List of `NERErrorComposite`
    """
    errors = []
    for doc in parser.docs:
        for sent in doc:
            if sent.ems_pairs:
                for ems_pair in sent.ems_pairs:
                    # overloading
                    try:  # funcs as a list
                        if boolean([is_func(ems_pair) for is_func in is_funcs]):
                            errors.append(ems_pair.result)
                    except TypeError:  # funcs as only a function
                        if is_funcs(ems_pair):
                            errors.append(ems_pair.result)
                    except NameError:
                        raise NameError('is_funcs is not one of the predefined result types')

    return errors

def get_false_errors_category(parser):
    _is_types = {'False Positive': is_fp, 'False Negative': is_fn, 'All False Errors': is_false_error}
    return {name: filtered_results(parser, is_func) for name, is_func in _is_types.items()}

def get_all_false_errors(parser):
    return filtered_results(parser, is_false_error)

def get_false_positives(parser):
    return filtered_results(parser, is_fp)

def get_false_negatives(parser):
    return filtered_results(parser, is_fn)

def get_span_errors_category(parser):
    _is_types = {'Right Expansion': is_re, 'Left Expansion': is_le, 'Right Left Expansion': is_rle,
                 'Right Diminished': is_rd, 'Left Diminished': is_ld, 'Right Left Diminished': is_rld,
                 'Right Crossed': is_rc, 'Left Crossed': is_lc,
                 'Spans Merged': is_merge, 'Span Split': is_split,
                 'Complicate': is_complicate_error,
                 'All Span Errors': is_span_error}
    return {name: filtered_results(parser, is_func) for name, is_func in _is_types.items()}

def get_all_span_errors(parser):
    return filtered_results(parser, is_span_error)

def get_only_span_errors(parser):
    return filtered_results(parser, is_only_span_error)

def get_all_type_errors(parser):
    return filtered_results(parser, is_type_error)

def get_only_type_errors(parser):
    return filtered_results(parser, is_only_type_error)

def get_span_and_type_composite_errors(parser):
    return filtered_results(parser, [is_type_error, is_span_error])

def get_only_span_and_type_composite_errors(parser):
    return filtered_results(parser, [is_type_error, is_span_error, lambda e: not is_complicate_error(e)])

def get_complicate_errors(parser):
    return filtered_results(parser, is_complicate_error)

def print_type_errors(parser):
    all_type_errors = get_all_type_errors(parser)

    error_table = {}

    for error in all_type_errors:
        type = str(error.type_error)
        if type not in error_table.keys():
            error_table[type] = 1
        else:
            error_table[type] += 1

    for key in sorted(error_table.keys()):
        print(key + ": " + str(error_table[key]))

def print_list_len_in_dict(dict_: Dict):
    for name, list_ in dict_.items():
        print('{}: {}'.format(name, len(list_)))


class NERErrorAnalyzer:
    """
    input: DocumentsWithErrorAnn
    """

    @classmethod
    def print_report(cls):
        pass

    @classmethod
    def save_report(cls, parser, tag_policy='conll'):

        fprint('Error Analysis')
        # print
        fnames = ['error_pie.png', 'false_error_heatmap.png', 'span_error_heatmap.png', 'confusion_matrix.png']
        cls.print_error_pie(parser, fnames[0])
        cls.print_false_error_heatmap(parser, tag_policy=tag_policy, save_file=fnames[1])
        cls.print_span_error_heatmap(parser, tag_policy=tag_policy, save_file=fnames[2])
        cls.pprint_ner_confusion_matrix(parser, tag_policy=tag_policy, save_file=fnames[3])

    @staticmethod
    def get_span_error_category_df(parser, transpose=True, tag_policy='conll'):
        errs = get_only_span_errors(parser)

        key1 = lambda err: err.span_error.type
        key2 = lambda err: err.ptypes[0]

        g = two_level_groupby(errs, key1, key2, count=True)
        df = DataFrame(g)
        if transpose:
            df = df.T

        span_etype_order = ['Right Expansion', 'Left Expansion', 'Right Left Expansion',
                            'Right Diminished', 'Left Diminished', 'Right Left Diminished',
                            'Right Crossed', 'Left Crossed',
                            'Spans Merged', 'Span Split',
                            'Complicate']

        if tag_policy == 'wnut':
            etypes = ['person', 'location', 'corporation', 'group', 'creative-work', 'product']
        elif tag_policy == 'conll':
            etypes = ['PER', 'LOC', 'ORG', 'MISC']

        df = sort_two_level_group(df, span_etype_order, etypes)
        add_total_column_row(df)

        return df



    @staticmethod
    def get_false_error_category_df(parser, transpose=True, tag_policy='conll'):
        errs = filtered_results(parser, is_false_error)

        key1 = lambda e: e.false_error.false_type
        key2 = lambda err: err.false_error.em_type
        g = two_level_groupby(errs, key1, key2, count=True)
        df = DataFrame(g)
        if transpose:
            df = df.T

        if tag_policy == 'wnut':
            etypes = ['person', 'location', 'corporation', 'group', 'creative-work', 'product']
        elif tag_policy == 'conll':
            etypes = ['PER', 'LOC', 'ORG', 'MISC']

        df = sort_two_level_group(df, ['False Positive', 'False Negative'], etypes)
        add_total_column_row(df)

        return df

    @staticmethod
    def print_error_pie(parser, save_file=None):
        only_span_errors = get_only_span_errors(parser)
        only_type_errors = get_only_type_errors(parser)
        only_span_and_type_composite_errors = get_only_span_and_type_composite_errors(parser)
        false_negatives = get_false_negatives(parser)
        false_positives = get_false_positives(parser)
        complicate_erorrs = get_complicate_errors(parser)
        # Pie chart
        labels = ['Only Span', 'Only Type', 'Span and Type', 'Complicate Errors', 'False Positive', 'False Negative']
        sizes = [len(only_span_errors), len(only_type_errors), len(only_span_and_type_composite_errors),
                 len(complicate_erorrs),
                 len(false_positives), len(false_negatives)]
        # colors
        # colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        fig1, ax1 = plt.subplots()

        ax1.pie(sizes, labels=labels, autopct=make_autopct(sizes), startangle=90)
        # draw circle
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')
        plt.tight_layout()

        if save_file is not None:
            plt.savefig(save_file)
        else:
            plt.show()

    @staticmethod
    def print_span_error_pie():  #TODO
        pass

    @staticmethod
    def print_composite_errors(parser):
        # get complicated error
        com_err_array = get_complicate_errors(parser)
        print(com_err_array)

    @staticmethod
    def print_false_errors(parser, tag_policy='conll'):  #TODO: duplicated?

        if tag_policy == 'wnut':
            etypes = ['person', 'location', 'corporation', 'group', 'creative-work', 'product']
        elif tag_policy == 'conll':
            etypes = ['PER', 'LOC', 'ORG', 'MISC']

        # group fp/fn by entity type
        fns = get_false_negatives(parser)
        fps = get_false_positives(parser)

        key = lambda err: err.false_error.em_type
        fn_groups = groupby(fns, key)
        fp_groups = groupby(fps, key)

        order = {key: i for i, key in enumerate(etypes)}
        for fp_type, fps in sorted(fp_groups.items(), key=lambda x: order[x[0]]):
            print('{} - {}'.format(fp_type, len(fps)))

        for fn_type, fns in sorted(fn_groups.items(), key=lambda x: order[x[0]]):
            print('{} - {}'.format(fn_type, len(fns)))

    @staticmethod
    def print_span_error_heatmap(parser, tag_policy='conll', save_file=None, **kwargs):
        err_df = NERErrorAnalyzer.get_span_error_category_df(parser, tag_policy=tag_policy)
        plt.figure()
        hm = sns.heatmap(err_df, annot=True, linewidth=1, fmt='.0f', cmap=cmap, **kwargs)
        hm.set_facecolor(np.append(cmap.colors[0][:-1], 0.5))
        hm.set_xticklabels(hm.get_xticklabels(), rotation=45)

        if save_file is not None:
            plt.savefig(save_file, bbox_inches='tight')
        else:
            plt.show()

    @staticmethod
    def print_false_error_heatmap(parser, save_file=None, tag_policy='conll', **kwargs):
        err_df = NERErrorAnalyzer.get_false_error_category_df(parser, tag_policy=tag_policy)
        plt.figure()
        hm = sns.heatmap(err_df, annot=True, linewidth=1, fmt='.0f', cmap=cmap, **kwargs)
        hm.set_facecolor(np.append(cmap.colors[0][:-1], 0.5))
        hm.set_xticklabels(hm.get_xticklabels(), rotation=45)
        hm.set_yticklabels(hm.get_yticklabels(), rotation=0)
        if save_file is not None:
            plt.savefig(save_file, bbox_inches='tight')
        else:
            plt.show()

    @classmethod
    def get_confusion_matrix_df(cls, parser: ConllParser, tag_policy='conll') -> DataFrame:

        wnut_types = ["person", "location", "corporation", "group", "creative-work", "product"]
        conll_types = ["PER", "LOC", "ORG", "MISC"]
        labels = wnut_types if tag_policy == 'wnut' else conll_types

        cm = cls.get_confusion_matrix(parser, tag_policy)

        return pd.DataFrame(cm, index=labels, columns=labels)

    @staticmethod
    def get_confusion_matrix(parser: ConllParser, tag_policy='conll') -> List[List[int]]:

        y_true = []
        y_pred = []

        for doc in parser.docs:
            for sentence in doc:
                if sentence.ems_pairs:
                    for ems_pair in sentence.ems_pairs.pairs:
                        if len(ems_pair.result.gtypes) == 1 and len(ems_pair.result.ptypes) == 1:
                            y_true.append(ems_pair.result.gtypes[0])
                            y_pred.append(ems_pair.result.ptypes[0])

        wnut_types = ["person", "location", "corporation", "group", "creative-work", "product"]
        conll_types = ["PER", "LOC", "ORG", "MISC"]
        labels = wnut_types if tag_policy == 'wnut' else conll_types

        return confusion_matrix(y_true, y_pred, labels=labels)

    @classmethod
    def print_ner_confusion_matrix(cls, parser, tag_policy='conll', **kwargs):
        cm = cls.get_confusion_matrix(parser, tag_policy=tag_policy)
        cls.print_confusion_matrix(cm, tag_policy=tag_policy, **kwargs)

    @classmethod
    def pprint_ner_confusion_matrix(cls, parser, tag_policy='conll', **kwargs):
        cm = cls.get_confusion_matrix(parser, tag_policy=tag_policy)
        cls.pprint_confusion_matrix(cm, tag_policy=tag_policy, **kwargs)

    @staticmethod
    def print_confusion_matrix(cm: List[List[int]], tag_policy='conll'):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)

        wnut_types = ["person", "location", "corporation", "group", "creative-work", "product"]
        conll_types = ["PER", "LOC", "ORG", "MISC"]
        labels = wnut_types if tag_policy == 'wnut' else conll_types

        plt.title('Confusion matrix of the classifier')
        fig.colorbar(cax)

        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        ax.xaxis.tick_bottom()
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

        plt.xlabel('Predicted')
        plt.ylabel('True')

        # Loop over data dimensions and create text annotations.
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="w")

        plt.show()

    @staticmethod
    def pprint_confusion_matrix(cm: List[List[int]], tag_policy='conll', **kwargs):

        wnut_types = ["person", "location", "corporation", "group", "creative-work", "product"]
        conll_types = ["PER", "LOC", "ORG", "MISC"]
        labels = wnut_types if tag_policy == 'wnut' else conll_types

        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        pretty_plot_confusion_matrix(df_cm, pred_val_axis='x', show_null_values=0, **kwargs)

if __name__ == '__main__':
    pass


