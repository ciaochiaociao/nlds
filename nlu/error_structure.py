from copy import copy
from typing import Tuple, Union, List, Dict, Optional

from ansi.color import fg

from nlu.data import TextList, EntityMentions, MD_IDs, Base
from nlu.utils import id_incrementer, overrides, colorize_list, ls_to_ls_str, TO_SEP, list_to_str

NOTE_SEP = '-'
NOTE = ' {} '.format(NOTE_SEP)


class EntityMentionsPair(TextList):
    """An EntityMentionsPair has a gold EntityMentions and a predicted EntityMentions
    >>> from nlu.data import Sentence, EntityMentions
    >>> sen = Sentence.from_str('NLU Lab is in Taipei Taiwan directed by Keh Yih Su .', 'I-ORG I-ORG O O I-LOC B-LOC O O I-PER I-PER I-PER O', 'I-ORG I-ORG O O I-LOC I-LOC O O O I-PER I-PER O')
    >>> pems = EntityMentions(sen.get_entity_mentions('predict')[1:3])
    >>> gems = EntityMentions(sen.get_entity_mentions('gold')[1:2])
    >>> error_id_inc = id_incrementer(), id_incrementer()
    >>> pair = EntityMentionsPair(0, gems, pems, error_id_inc)
    >>> pair  # doctest: +ELLIPSIS
    EntityMentionsPair(...)
    >>> len(pair.gems), len(pair.pems), len(pair.ems)
    (1, 2, 3)
    >>> print(pair)
    D99-S999-PAIR0: (G) NLU Lab is in \x1b[33m[Taipei Taiwan]LOC\x1b[0m directed by Keh Yih Su . (P) NLU Lab is in \x1b[34m[Taipei]LOC\x1b[0m \x1b[34m[Taiwan]LOC\x1b[0m directed by Keh Yih Su .
    >>> any(em.entity_mentions_pair is not pair for em in pair.gems.mentions)
    False
    >>> pair.pprint()
    --- error id D99-S999-PAIR0 ---
    (G) NLU Lab is in \x1b[33m[Taipei Taiwan]LOC\x1b[0m directed by Keh Yih Su .
    (P) NLU Lab is in \x1b[34m[Taipei]LOC\x1b[0m \x1b[34m[Taiwan]LOC\x1b[0m directed by Keh Yih Su .
    """
    def __init__(self, id_, gems: EntityMentions, pems: EntityMentions, id_incs: Tuple[id_incrementer, id_incrementer]):  
        # TODO: idea - Set A Proxy Class for List[EntityMention] like EntityMentions (like `view`)
        # TODO: fix id_incs generator
        # TODO: Use List[EntityMention] instead of EntityMentions to be input into TextList constructor
        self.gems = gems
        self.pems = pems
        self.mentions: EntityMentions = self.pems + self.gems
        self.ems: EntityMentions = self.mentions
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

        TextList.__init__(self, ids, self.mentions.mentions)
        
        # navigation
        self.sentence = self.mentions[0].sentence

        self.sid = self.ids['S']
        self.did = self.ids['D']
        
        # set backreference to entity mentions
#         self.set_ems_pair_to_ems()

    def __str__(self):

        em = self.pems[0] if len(self.pems.members) > 0 else self.gems[0]
        ids = MD_IDs.from_list([('D', em.did), ('S', em.sid), ('PAIR', self.ids['PAIR'])])

        gem_sent = self.sentence.get_ann_sent(self.gems, fg.yellow) if len(self.gems) > 0 else str(self.sentence)
        pem_sent = self.sentence.get_ann_sent(self.pems) if len(self.pems) > 0 else str(self.sentence)

        return ids.fullid + ': (G) ' + gem_sent + ' (P) ' + pem_sent
        
    def pprint(self) -> None:

        print('--- error id %s ---' % self.fullid)
        print('(G)', self.sentence.get_ann_sent(self.gems, fg.yellow))
        print('(P)', self.sentence.get_ann_sent(self.pems))


    @overrides(TextList)
    def __add__(self, other):
        return EntityMentionsPair(self.id, self.gems + other.gems, self.pems + other.pems, self.id_incs)
    
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
            
#     def set_ems_pair_to_ems(self) -> None:
#         for em in self.mentions:
#             em.ems_pair = self


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
        self.correct_id = correct_id

    def __str__(self):
        return '---{}---'.format(fg.green('Correct')) + NERComparison.__str__(self)

    
class NERError(Base):
    pass

class SpanError(NERError):
    """
    >>> from nlu.data import ConllToken, ConllNERTag, Sentence, EntityMention
    >>> from nlu.error import NERErrorAnnotator, NERErrorExtractor
    >>> from nlu.data import Sentence, EntityMentions
    >>> sen = Sentence.from_str('NLU Lab is in Taipei Taiwan directed by Keh Yih Su .', 'I-ORG I-ORG O O I-LOC B-LOC O O I-PER I-PER I-PER O', 'I-ORG I-ORG O O I-LOC I-LOC O O O I-PER I-PER O')
    >>> sen.set_entity_mentions()
    >>> len(sen.gems), len(sen.pems)
    (3, 4)
    >>> pair = EntityMentionsPair(0, EntityMentions(sen.gems[2:3]), EntityMentions(sen.pems[3:4]), id_incrementer())
    >>> print(str(SpanError(pair, 'R', 'Diminished')))
    R Diminished
    """
    def __init__(self, ems_pair: EntityMentionsPair, direction, span_type):

        self.direction: str = direction
        self.span_type: str = span_type
        self.ems_pair = ems_pair
        self.type = direction + ' ' + span_type

    def __str__(self):
        return self.direction + ' ' + self.span_type


class MergeSplitError(SpanError):

    def __init__(self, ems_pair: EntityMentionsPair, type_):
        self.ems_pair = ems_pair
        self.type = type_
        self.type_ = type_
        self.gems = ems_pair.gems
        self.pems = ems_pair.pems

    def __str__(self):
        return self.name_str()

    def name_str(self, sep=None):
        return ls_to_ls_str(self.gems, self.pems, sep=sep)


class FalseError(NERError):
    def __init__(self, ems_pair: EntityMentionsPair, false_type):
        self.false_type: str = false_type
        self.em_type: str = ems_pair.mentions.type
        self.ems_pair = ems_pair

    def __str__(self):
        return self.false_type + NOTE + self.em_type


class MentionTypeError(NERError):
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


class ComplicateError(SpanError):
    """Complicated Case:
        - [O][OO] <-> [OO][O] both pems and gems have the same range +
        but none of them have only one entity mention
        - [OO]O[O] <-> [OOOO] not concat
        - [OO][O] <-> [O][OOO] not same range
    """
    def __init__(self, ems_pair: EntityMentionsPair):
        self.ems_pair = ems_pair
        self.predict_types: List[str] = ems_pair.pems.types
        self.gold_types: List[str] = ems_pair.gems.types
        self.type = 'Complicate'

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

    def __init__(self, ems_pair: EntityMentionsPair, 
                 type: Dict[str, Union[str, SpanError, MentionTypeError, MergeSplitError, ComplicateError]], 
                 error_id):
        ids = copy(ems_pair.parent_ids)
        ids.update({'NERErr': next(error_id)})
        super().__init__(ems_pair, ids)
        self.type = type
        #TODO: type of error directly saved
        # if is_complicate_error(self.span_error):
        #     self.type2 = 'Span Error'
        # elif is_fp(self.false_error):
        #     self.type2 = 'False Positive'
        # elif is_fn(self.false_error):
        #     self.type2 = 'False Negative'
        # elif is_span_error(self.span_error) and
        #
        self.error_id = error_id
        self.filtered_type = self.filtered_to_array(type)
        self.false_error = self.type['false_error']
        self.span_error: SpanError = self.type['span_error']
        self.type_error = self.type['type_error']
        self.ptext = self.pems.text
        self.gtext = self.gems.text

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

def is_error(ems_pair):
    return isinstance(ems_pair.result, NERErrorComposite)

def is_correct(ems_pair):
    return isinstance(ems_pair.result, NERCorrect)

def is_span_error(ems_pair):
    return is_error(ems_pair) and isinstance(ems_pair.result.span_error, SpanError)

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
    return is_error(ems_pair) and isinstance(ems_pair.result.span_error, ComplicateError)


if __name__ == '__main__':
    import doctest

    failure_count, test_count = doctest.testmod()
    
    if failure_count == 0:
        
        print('{} tests passed!!!!!!'.format(test_count))
    