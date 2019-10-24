from doctest import testmod

from nlu.error import NERErrorAnnotator as ntr
from nlu.parser import *
from nlu.error_analysis import NERErrorAnalyzer as nr

# basic doctest
testmod()

# wnut
cols_format = [{'type': 'gold', 'col_num': 1, 'tagger': 'ner'},
                {'type': 'predict', 'col_num': 2, 'tagger': 'ner'},
                {'type': 'predict', 'col_num': 3, 'tagger': 'ner_conf'}]
parser = ConllParser('test/wnut.test.gold.pred.iob1', cols_format)
print(parser.docs[0][0][0].conf)
parser.set_entity_mentions(tag_policy='wnut')
parser.obtain_statistics(entity_stat=True, source='predict', tag_policy='wnut')
parser.obtain_statistics(entity_stat=True, source='gold', tag_policy='wnut')
ntr.annotate(parser)
parser.print_n_corrects(50)
parser.print_n_errors(10)
parser.error_overall_stats()
nr.save_report(parser, tag_policy='wnut')