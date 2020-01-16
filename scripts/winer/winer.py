from doctest import testmod

from nlu.error import *
from nlu.parser import *
from nlu.error_analysis import *

# basic doctest
# testmod()

# conll
cols_format = [{'type': 'gold', 'col_num': 1, 'tagger': 'ner'},
                {'type': 'predict', 'col_num': 2, 'tagger': 'ner'},
               {'type': 'predict', 'col_num': 3, 'tagger': 'ner_conf'}]

parser = ConllParser('final_finer_100000.gold.pred', cols_format, tag_scheme='iob2')
print(parser.docs[0][0][0].conf)
parser.set_entity_mentions()
parser.obtain_statistics(entity_stat=True, source='predict')
parser.obtain_statistics(entity_stat=True, source='gold')
NERErrorAnnotator.annotate(parser)
parser.print_n_corrects(50)
parser.print_n_errors(10)
parser.error_overall_stats()
cm = NERErrorAnalyzer.get_confusion_matrix(parser)
# NERErrorAnalyzer.print_confusion_matrix(cm)
NERErrorAnalyzer.pprint_confusion_matrix(cm, save_file='winer_workspace/confusion_matrix_conll.png')
NERErrorAnalyzer.save_report(parser)
parser.save_result('winer.tsv')