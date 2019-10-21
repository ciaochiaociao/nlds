from doctest import testmod

from nlu.error import *
from nlu.parser import *

# basic doctest
testmod()

cols_format = [{'type': 'gold', 'col_num': 1, 'tagger': 'ner'},
                {'type': 'predict', 'col_num': 2, 'tagger': 'ner'},
                {'type': 'predict', 'col_num': 3, 'tagger': 'ner_conf'}]

parser = ConllParser('test/wnut.test.gold.pred.iob1', cols_format)
print(parser.docs[0][0][0].conf)

parser.set_entity_mentions(tag_policy='wnut')

parser.obtain_statistics(entity_stat=True, source='predict')


parser.obtain_statistics(entity_stat=True, source='gold')

NERErrorAnnotator.annotate(parser)

parser.print_corrects()

parser.print_all_errors()

parser.error_overall_stats()

