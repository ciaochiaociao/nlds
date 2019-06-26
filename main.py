from doctest import testmod

from nlu.error import *
from nlu.parser import *

# basic doctest
testmod()

cols_format = [{'type': 'predict', 'col_num': 1, 'tagger': 'ner'},
                {'type': 'gold', 'col_num': 2, 'tagger': 'ner'}]

parser = ConllParser('rcv1.testb.compare2', cols_format)

parser.obtain_statistics(entity_stat=True, source='predict')

parser.obtain_statistics(entity_stat=True, source='gold')

NERErrorAnnotator.annotate(parser)

parser.print_corrects()

parser.print_all_errors()

parser.error_overall_stats()

