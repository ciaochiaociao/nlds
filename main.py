from doctest import testmod

from nlu.error import *
from nlu.parser import *

# basic doctest
testmod()

train_parser = ConllParser('test.txt')

train_parser.obtain_statistics(entity_stat=True, source='predict')

train_parser.obtain_statistics(entity_stat=True, source='gold')

ParserNERErrors(train_parser)

total = 0
for doc in train_parser.docs:
    for sentence in doc:
        total += 1
        if sentence.em_overlaps:
            for res in sentence.em_overlaps.results:
                print(str(res))

print(total)