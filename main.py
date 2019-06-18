from nlu.error import *
from nlu.parser import *

# basic doctest
doctest.testmod()

train_parser = ConllParser('rcv1.train.compare2')

train_parser.obtain_statistics(entity_stat=True, source='predict')

train_parser.obtain_statistics(entity_stat=True, source='gold')

ParserNERErrors(train_parser)

