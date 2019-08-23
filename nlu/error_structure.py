from copy import copy
from typing import Tuple, Union, List, Dict, Optional

from ansi.color import fg

from nlu.data import TextList, EntityMentions, MD_IDs
from nlu.utils import id_incrementer, overrides, colorize_list, ls_to_ls_str, TO_SEP

NOTE_SEP = '-'
NOTE = ' {} '.format(NOTE_SEP)


class EntityMentionsPair(TextList):
    """An EntityMentionsPair has a gold EntityMentions and a predicted EntityMentions
    >>> from nlu.data import ConllToken, ConllNERTag, Sentence, EntityMention
    >>> from nlu.error import NERErrorAnnotator, NERErrorExtractor
    >>> sid, did = 3, 50
    >>> taiwan = ConllToken('Taiwan', 0, sid, did, ners={'gold':ConllNERTag('B-ORG'), 'predict':ConllNERTag('I-LOC')})
    >>> semi = ConllToken('Semiconductor', 1, sid, did,\
    ners={'gold':ConllNERTag('I-ORG'), 'predict':ConllNERTag('I-MISC')})
    >>> manu = ConllToken('Manufacturer', 2, sid, did,\
    ners={'gold':ConllNERTag('I-ORG'), 'predict':ConllNERTag('I-ORG')})
    >>> co = ConllToken('Cooperation', 3, sid, did, ners={'gold':ConllNERTag('I-ORG'), 'predict':ConllNERTag('I-ORG')})
    >>> sen = Sentence([taiwan, semi, manu])
    >>> for token in [taiwan, semi, manu, co]: \
            token.set_sentence(sen)
    >>> tsmc = EntityMention([taiwan, semi, manu], source='gold', id_=0)
    >>> t = EntityMention([taiwan], source='predict', id_=0)
    >>> tsmcs = EntityMentions([tsmc])
    >>> ts = EntityMentions([t])
    >>> error_id_inc = id_incrementer(), id_incrementer()
    >>> pair = EntityMentionsPair(0, tsmcs, ts, error_id_inc)
    >>> print(pair)
    Taiwan Taiwan Semiconductor Manufacturer
    >>> print(pair.fullid)
    D50-S3-PAIR0
    """
    def __init__(self, id_, gems: EntityMentions, pems: EntityMentions, id_incs: Tuple[id_incrementer, id_incrementer]):
        self.gems = gems
        self.pems = pems
        self.mentions = self.pems + self.gems
        self.id_incs = id_incs
        self.result: Optional[NERComparison] = None
        self.correct: Optional[NERCorrect] = None
        self.error: Optional[NERErrorComposite] = None

        # ids
        try:
            self.sid, self.did = self.mentions[0].sid, self.mentions[0].did
        except IndexError:
            raise IndexError("Can't access the index of gold mentions and predicted mentions! Probably there is no any "
                             "mentions in the ems_pair group")

        self.source = self.mentions[0].source

        ids = copy(self.mentions.ids)
        ids.update({'PAIR': id_})

        # number
        self.pems_total = len(pems)
        self.gems_total = len(gems)
        self.ems_total = len(pems + gems)

        self.token_b = min([em.token_b for em in self.mentions])
        self.token_e = max([em.token_e for em in self.mentions])

        # text
        self.predict_text = ' '.join([str(pem) for pem in pems])
        self.gold_text = ' '.join([str(gem) for gem in gems])
        self.predict_text_sep = '|'.join([str(pem) for pem in pems])
        self.gold_text_sep = '|'.join([str(gem) for gem in gems])

        TextList.__init__(self, ids, self.mentions)

        # navigation
        self.sentence = self.mentions[0].sentence

        self.sid = self.ids['S']
        self.did = self.ids['D']

    def pretty_print(self) -> None:
        print('--- error id %s ---' % self.fullid)
        print('< gold >')
        for gem in self.gems:
            print('{} ({}-{}) '.format(gem.text, gem.token_b, gem.token_e))
        print('< predict >')
        for pem in self.pems:
            print('{} ({}-{}) '.format(pem.text, pem.token_b, pem.token_e))

    @overrides(TextList)
    def __add__(self, other):
        return EntityMentionsPair(self.id, self.gems + other.gems, self.pems + other.pems, self.id_incs)
    #
    # def ann_str(self) -> str:
    #     result = []
    #     for i in range(self.token_b, self.token_e+1):
    #         if i in self.pems.token_bs:
    #             result.append('[' + self.)
    #
    #     return [pem for pem in self.pems]

    def set_result(self, result) -> None:
        self.result: Union[NERErrorComposite, NERCorrect] = result
        self.set_error()
        self.set_correct()

    def set_error(self) -> None:
        try:
            self.error: Union[None, NERErrorComposite] = self.result if not self.iscorrect() else None
        except AttributeError:
            raise AttributeError('Use {} first to set the result of the EntityMentionsPair'.format(self.set_result.__name__))

    def set_correct(self) -> None:
        try:
            self.correct: Union[None, NERCorrect] = self.result if self.iscorrect() else None
        except AttributeError:
            raise AttributeError('Use {} first to set the result of the EntityMentionsPair'.format(self.set_result.__name__))

    def iscorrect(self) -> bool:
        if isinstance(self.result, NERCorrect):
            return True
        elif isinstance(self.result, NERErrorComposite):
            return False
        else:
            raise TypeError('The returned result is neither {} nor {}, but {} is obtained'
                            .format(NERCorrect.__name__, NERErrorComposite.__name__, type(self.result)))


class EntityMentionsPairs(TextList):
    """One sentence will have one EntityMentionsPairs"""
    def __init__(self, pairs: List[EntityMentionsPair]):
        self.pairs = pairs
        self.results: Optional[List[NERComparison]] = None
        self.corrects: Optional[List[NERCorrect]] = None
        self.errors: Optional[List[NERErrorComposite]] = None

        try:
            ids = self.pairs[0].ids
        except IndexError:
            raise IndexError("""Can't access the first element of the pairs. Overlaps should be empty.
            repr(pairs): {}
            """.format(repr(self.pairs)))

        TextList.__init__(self, ids, self.pairs)


class NERComparison(MD_IDs):

    entity_types = ('PER', 'LOC', 'ORG', 'MISC')

    def __init__(self, ems_pair: EntityMentionsPair, ids):

        self.ems_pair = ems_pair
        self.sentence = self.ems_pair.sentence
        self.gems = ems_pair.gems
        self.pems = ems_pair.pems
        self.gems_total = len(self.gems)
        self.pems_total = len(self.pems)
        self.gtypes = [str(gem.type) for gem in self.gems]
        self.ptypes = [str(pem.type) for pem in self.pems]
        self.ems_total = self.gems_total + self.pems_total
        self.filtered_type = None
        MD_IDs.__init__(self, ids)

    def __str__(self):

        return '\n[predict] {} ({})'.format(self.pems.sep_str(sep='|'), self.ptypes ) + \
               '\n[gold] {} ({})'.format(self.gems.sep_str(sep='|'), self.gtypes) + \
               '\n[type] {}'.format(str(self.filtered_type)) + \
               '\n[sentence] {}'.format(colorize_list(
                   self.sentence.tokens, self.ems_pair.token_b, self.ems_pair.token_e)) + \
               '\n[ID] {}'.format(self.fullid) + \
               '\n'
        # self.type - use type of NERErrorComposite and NERCorrect


class NERCorrect(NERComparison):

    def __init__(self, ems_pair: EntityMentionsPair, type: str, correct_id):
        ids = copy(ems_pair.parent_ids)
        ids.update({'NERCorr': next(correct_id)})
        super().__init__(ems_pair, ids)
        self.type = type
        self.filtered_type = type

    def __str__(self):
        return '---{}---'.format(fg.green('Correct')) + NERComparison.__str__(self)


class SpanError:
    """
    >>> from nlu.data import ConllToken, ConllNERTag, Sentence, EntityMention
    >>> from nlu.error import NERErrorAnnotator, NERErrorExtractor
    >>> sid, did = 3, 50
    >>> taiwan = ConllToken('Taiwan', 0, sid, did, ners={'gold':ConllNERTag('B-ORG'), 'predict':ConllNERTag('I-LOC')})
    >>> semi = ConllToken('Semiconductor', 1, sid, did,\
    ners={'gold':ConllNERTag('I-ORG'), 'predict':ConllNERTag('I-MISC')})
    >>> manu = ConllToken('Manufacturer', 2, sid, did,\
    ners={'gold':ConllNERTag('I-ORG'), 'predict':ConllNERTag('I-ORG')})
    >>> co = ConllToken('Cooperation', 3, sid, did, ners={'gold':ConllNERTag('I-ORG'), 'predict':ConllNERTag('I-ORG')})
    >>> sen = Sentence([taiwan, semi, manu])
    >>> for token in [taiwan, semi, manu, co]: \
            token.set_sentence(sen)
    >>> tsmc = EntityMention([taiwan, semi, manu], source='gold', id_=0)
    >>> t = EntityMention([taiwan], source='predict', id_=0)
    >>> tsmcs = EntityMentions([tsmc])
    >>> ts = EntityMentions([t])
    >>> error_id_inc = id_incrementer(), id_incrementer()
    >>> pair = EntityMentionsPair(0, tsmcs, ts, error_id_inc)
    >>> print(str(SpanError(pair, 'R', 'Expansion')))
    1
    """
    def __init__(self, ems_pair: EntityMentionsPair, direction, span_type):

        self.direction: str = direction
        self.span_type: str = span_type
        self.ems_pair = ems_pair

    def __str__(self):
        return self.direction + ' ' + self.span_type


class MergeSplitError:

    def __init__(self, ems_pair: EntityMentionsPair, type_):
        self.ems_pair = ems_pair
        self.type = type_
        self.gems = ems_pair.gems
        self.pems = ems_pair.pems

    def __str__(self):
        return self.name_str() + str(self.ems_pair.ems_total)

    def name_str(self, sep=None):
        return ls_to_ls_str(self.gems, self.pems, sep=sep)


class FalseError:
    def __init__(self, ems_pair: EntityMentionsPair, false_type):
        self.false_type: str = false_type
        self.em_type: str = ems_pair.mentions.type
        self.ems_pair = ems_pair

    def __str__(self):
        return self.false_type + NOTE + self.em_type


class MentionTypeError:
    def __init__(self, ems_pair: EntityMentionsPair):
        self.ems_pair = ems_pair
        self.gems = ems_pair.gems
        self.pems = ems_pair.pems
        self.gold_types: List[str] = self.gems.types
        self.predict_types: List[str] = self.pems.types

    def __str__(self):
        return self.type_str()

    def type_str(self, sep=None):
        return ls_to_ls_str(self.gems.types, self.pems.types, sep=sep)


class ComplicateError:
    def __init__(self, ems_pair: EntityMentionsPair):
        self.ems_pair = ems_pair
        self.predict_types: List[str] = ems_pair.pems.types
        self.gold_types: List[str] = ems_pair.gems.types

    def __str__(self):
        return 'Complicate - ' + self.ems_pair.gold_text_sep + TO_SEP + self.ems_pair.predict_text_sep


class NERErrorComposite(NERComparison):

    type_error_types, false_error_types = [], []

    for t1 in NERComparison.entity_types:
        for t2 in NERComparison.entity_types:
            if t1 != t2:
                type_error_types.append(t1 + '->' + t2)
        for f in ['False Positive', 'False Negative']:
            false_error_types.append(f + ' - ' + t1)

    span_errors1 = ['R Expansion', 'L Expansion', 'RL Expansion',
                    'R Diminished', 'L Diminished', 'RL Diminished',
                    'R Crossed', 'L Crossed']

    span_errors2 = ['Spans Merged', 'Span Split', 'Complicated']

    all_span_error_types = span_errors1 + span_errors2

    all_types = type_error_types + all_span_error_types

    # {'type': str, 'pems': EntityMentions, 'gems': EntityMentions, ''}
    # {'false_error': false_error, 'type_error': type_error, 'span_error': span_error}

    def __init__(self, ems_pair: EntityMentionsPair, type: Dict[str, Union[str, SpanError, MentionTypeError
    , MergeSplitError, ComplicateError]], error_id):
        ids = copy(ems_pair.parent_ids)
        ids.update({'NERErr': next(error_id)})
        super().__init__(ems_pair, ids)
        self.type = type
        self.filtered_type = self.filtered_to_array(type)
        self.false_error = self.type['false_error']
        self.span_error: SpanError = self.type['span_error']
        self.type_error = self.type['type_error']

    def __str__(self):
        return '---{}---'.format(fg.red('Error')) + NERComparison.__str__(self)

    @staticmethod
    def filtered_to_array(type_) -> List:
        return list(NERErrorComposite.filtered_err_str_list(type_).values())

    @staticmethod
    def filtered_err_str_list(type_: Dict[str, Union[str, int]]) -> Dict:
        return {error_cat:str(t) for error_cat, t in type_.items() if t is not None}

    def ann_str(self) -> str:
        return ''

    # @property  # todo
    # def type(self):
    #     return self.__type
    #
    # @type.setter
    # def type(self, type: Dict[str, Union[str, int]]):
    #
    #     if type['false_error'] in NERErrorComposite.false_error_types + [None]\
    #         and type['type_error'] in NERErrorComposite.type_error_types + [None]\
    #             and type['span_error'] in NERErrorComposite.all_span_error_types + [None]:
    #         self.__type = type
    #     else:
    #         raise ValueError("""The type '{}' is not in one of these:
    #         '{} or None'
    #         """.format(type, NERErrorComposite.all_types))
