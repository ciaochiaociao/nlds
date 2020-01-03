from doctest import testmod

from nlu.error import *
from nlu.parser import *
from nlu.error_analysis import *
from nlu.error import NERErrorAnnotator as ntr
from nlu.error_analysis import NERErrorAnalyzer as nr

# basic doctest
# testmod()

# conll
cols_format = [{'type': 'gold', 'col_num': 1, 'tagger': 'ner'},
                {'type': 'predict', 'col_num': 2, 'tagger': 'ner'},
               {'type': 'predict', 'col_num': 3, 'tagger': 'ner_conf'}]
parser = ConllParser('test/testb_new.gold.pred', cols_format, tag_scheme='iob1')
print(parser.docs[0][0][0].conf)
parser.set_entity_mentions(tag_policy='conll')
parser.obtain_statistics(entity_stat=True, source='predict')
parser.obtain_statistics(entity_stat=True, source='gold')
ntr.annotate(parser)
parser.print_n_corrects(50)
parser.print_n_errors(10)
parser.error_overall_stats()
cm = nr.get_confusion_matrix(parser)
# NERErrorAnalyzer.print_confusion_matrix(cm)
nr.pprint_confusion_matrix(cm, save_file='confusion_matrix_conll.png')
nr.save_report(parser, tag_policy='conll')
parser.save_result('conll_test_result.tsv')
