from typing import Tuple

from ansi.color import fg

from nlu.data import *
from nlu.parser import ConllParser
from nlu.utils import *


class EntityMentionsPair(TextList):
    """An EntityMentionsPair has a gold EntityMentions and a predicted EntityMentions
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
    D50-S3-OL0
    >>> pair.compare()
    >>> print(pair.result)
    R Diminished None ORG->LOC
    >>> print(str(pair.result.type))
    {'false_error': None, 'type_error': 'ORG->LOC', 'span_error': 'R Diminished'}
    """
    def __init__(self, id_, gems: EntityMentions, pems: EntityMentions, id_incs: Tuple[id_incrementer, id_incrementer]):
        self.gems = gems
        self.pems = pems
        self.mentions = self.pems + self.gems
        self.id_incs = id_incs

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

        self.compare()

    def pretty_print(self) -> None:
        print('--- error id %i ---' % self.fullid)
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

    def compare(self):
        self.result: Union[NERError, NERCorrect] = NERAnalyzer.extract(self, self.id_incs)  # TODO: move to parser class
        if self.iscorrect():
            self.correct: Union[None, NERCorrect] = self.result
            self.error: Union[None, NERError] = None
        else:
            self.error: Union[None, NERError] = self.result
            self.correct: Union[None, NERCorrect] = None

    def iscorrect(self):
        if isinstance(self.result, NERCorrect):
            return True
        elif isinstance(self.result, NERError):
            return False
        else:
            raise TypeError('The returned result is neither {} nor {}', NERCorrect.__name__, NERError.__name__)


class EntityMentionsPairs(TextList):
    """One sentence will have one EntityMentionsPairs"""
    def __init__(self, pairs: List[EntityMentionsPair]):
        self.pairs = pairs
        self.results = [pair.result for pair in pairs]
        self.corrects = [pair.correct for pair in pairs if pair.correct is not None]
        self.errors = [pair.error for pair in pairs if pair.error is not None]

        try:
            ids = self.pairs[0].ids
        except IndexError:
            raise IndexError("""Can't access the first element of the pairs. Overlaps should be empty.
            repr(pairs): {}
            """.format(repr(self.pairs)))

        TextList.__init__(self, ids, self.pairs)


class NERComparison(MD_IDs):

    entity_types = ('PER', 'LOC', 'ORG', 'MISC')

    def __init__(self, em_pair: EntityMentionsPair, ids):

        self.ems_pair = em_pair
        self.sentence = self.ems_pair.sentence
        self.gems = em_pair.gems
        self.pems = em_pair.pems
        self.gems_total = len(self.gems)
        self.pems_total = len(self.pems)
        self.gtypes = [str(gem.type) for gem in self.gems]
        self.ptypes = [str(pem.type) for pem in self.pems]
        self.ems_total = self.gems_total + self.pems_total
        MD_IDs.__init__(self, ids)

    def __str__(self):

        return '\n[predict] {} ({})'.format(self.pems.sep_str(sep='|'), self.ptypes ) + \
               '\n[gold] {} ({})'.format(self.gems.sep_str(sep='|'), self.gtypes) + \
               '\n[type] {}'.format(str(self.filtered_type)) + \
               '\n[sentence] {}'.format(colorize_list(
                   self.sentence.tokens, self.ems_pair.token_b, self.ems_pair.token_e)) + \
               '\n[ID] {}'.format(self.fullid) + \
               '\n'
        # self.type - use type of NERError and NERCorrect


class NERCorrect(NERComparison):

    def __init__(self, em_pair: EntityMentionsPair, type: str, correct_id):
        ids = copy(em_pair.parent_ids)
        ids.update({'NERCorr': next(correct_id)})
        super().__init__(em_pair, ids)
        self.type = type
        self.filtered_type = type

    def __str__(self):
        return '---{}---'.format(fg.green('Correct')) + NERComparison.__str__(self)


class NERError(NERComparison):
    """
    """

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

    def __init__(self, em_pair: EntityMentionsPair, type: Dict[str, Union[str, int]], error_id):
        ids = copy(em_pair.parent_ids)
        ids.update({'NERErr': next(error_id)})  # FIXME
        super().__init__(em_pair, ids)
        self.type = type
        self.filtered_type = NERError.filtered_to_array(type)
        self.false_error = self.type['false_error']
        self.span_error = self.type['span_error']
        self.type_error = self.type['type_error']

    def __str__(self):
        return '---{}---'.format(fg.red('Error')) + NERComparison.__str__(self)

    @staticmethod
    def filtered_to_array(type) -> List:
        return list(NERError.filtered(type).values())

    @staticmethod
    def filtered(type: Dict[str, Union[str, int]]) -> Dict:
        return {error_cat:type_ for error_cat, type_ in type.items() if type_ is not None}

    def ann_str(self) -> str:
        return
    # def is_false_positive(self):  # TODO
    #     return self.false_error in
    #
    # def set_different_errors(self):
    #     if self.false_error in
    #     self.

    # @property  # todo
    # def type(self):
    #     return self.__type
    #
    # @type.setter
    # def type(self, type: Dict[str, Union[str, int]]):
    #
    #     if type['false_error'] in NERError.false_error_types + [None]\
    #         and type['type_error'] in NERError.type_error_types + [None]\
    #             and type['span_error'] in NERError.all_span_error_types + [None]:
    #         self.__type = type
    #     else:
    #         raise ValueError("""The type '{}' is not in one of these:
    #         '{} or None'
    #         """.format(type, NERError.all_types))


class NERAnalyzer:
    """
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
    >>> id_incs = id_incrementer(), id_incrementer()
    >>> pair = EntityMentionsPair(0, tsmcs, ts, id_incs)
    >>> NERAnalyzer.extract(pair, id_incs).type
    {'false_error': None, 'type_error': 'ORG->LOC', 'span_error': 'R Diminished'}
    >>> co_pem = EntityMention([co], source='predict', id_=3)  # test NERCorrect
    >>> co_gem = EntityMention([co], source='gold', id_=3)
    >>> pair_correct = EntityMentionsPair(1, EntityMentions([co_pem]), EntityMentions([co_gem]), id_incs)
    >>> repr(NERAnalyzer.extract(pair_correct, id_incs))  # doctest: +ELLIPSIS
    '<error.NERCorrect object at ...
    >>> NERAnalyzer.extract(pair_correct, id_incs).type
    'ORG'
    """

    @staticmethod
    def extract(em_pair: EntityMentionsPair, id_incs: Tuple[id_incrementer, id_incrementer]) \
            -> Union[NERError, NERCorrect]:
        false_error = None
        type_error = None
        span_error = None
        pems: EntityMentions = em_pair.pems
        gems: EntityMentions = em_pair.gems
        ems_total = len(gems) + len(pems)

        error_id, correct_id = id_incs

        if NERAnalyzer.is_mentions_correct(pems, gems):
            return NERCorrect(em_pair, em_pair.gems.type, correct_id)
        else:
            if ems_total == 1:  # False Positive / False Negative
                false_error = 'False Negative - ' + gems[0].type if len(gems) == 1 else 'False Positive - ' + pems[0].type

            elif ems_total == 2:  # Span Error / Type Errors (type_errors)
                pb, pe, gb, ge = pems[0].token_b, pems[0].token_e, gems[0].token_b, gems[0].token_e
                pt, gt = pems[0].type, gems[0].type

                if pb == gb and pe != ge:
                    span_error = 'R Expansion' if pe > ge else 'R Diminished'
                elif pb != gb and pe == ge:
                    span_error = 'L Expansion' if pb < gb else 'L Diminished'
                elif pb != gb and pe != ge:
                    if pb < gb:
                        span_error = 'L Crossed' if pe < ge else 'RL Expansion'
                    elif pb > gb:
                        span_error = 'R Crossed' if pe > ge else 'RL Diminished'

                if pt != gt:  # Type Errors (type_errors)
                    type_error = gems[0].type + ' -> ' + pems[0].type

            elif ems_total >= 3:

                if NERAnalyzer.is_concatenated(pems) and NERAnalyzer.is_concatenated(gems):

                    if NERAnalyzer.has_same_range(pems, gems):
                        if len(pems) == 1:
                            span_error = 'Spans Merged - ' + str(len(gems))
                        elif len(gems) == 1:
                            span_error = 'Span Split - ' + str(len(pems))

                    if not NERAnalyzer.has_same_type(pems, gems):
                        type_error = '|'.join(gems.types) + ' -> ' + '|'.join(pems.types)
                    # input('Merge/Split: {}->{}'.format(self.predict_text, self.gold_text))

                else:
                    span_error = 'Complicated - {}->{}'.format(len(gems), len(pems))
                    print('Complicated Case:', [(pem.token_b, pem.token_e, pem.type) for pem in pems],
                          [(gem.token_b, gem.token_e, gem.type) for gem in gems],
                          NERAnalyzer.is_mentions_correct(gems, pems))

        return NERError(em_pair, {'false_error': false_error, 'type_error': type_error,
                                  'span_error': span_error}, error_id)

    @staticmethod
    def has_same_type(ems1: List[EntityMention], ems2: List[EntityMention]) -> bool:

        pts = {em1.type for em1 in ems1}
        gts = {em2.type for em2 in ems2}

        return pts == gts

    @staticmethod
    def has_same_range(ems1: List[EntityMention], ems2: List[EntityMention]) -> bool:
        """return if two lists of EntityMentions have the same 'whole range'
        (The whole range here is from the position of the first token to the position of the last token of the list.)
        """
        return (ems1[0].token_b, ems1[-1].token_e) == (ems2[0].token_b, ems2[-1].token_e)

    @staticmethod
    def is_mentions_correct(gems: EntityMentions, pems: EntityMentions) -> bool:
        # Handle different number of gold mentions and predicted mentions
        if len(pems) != len(gems):
            return False
        for pem, gem in zip(pems, gems):
            if not NERAnalyzer.is_mention_correct(pem, gem):
                return False
        return True

    @staticmethod
    def is_mention_correct(gold_em: EntityMention, predict_em: EntityMention) -> bool:
        return (predict_em.token_b, predict_em.token_e, predict_em.type) == \
               (gold_em.token_b, gold_em.token_e, gold_em.type)

    @staticmethod
    def is_concatenated(mentions: List[EntityMention]):
        last_em = None
        for i, em in enumerate(mentions):
            if last_em is not None and em.token_b - last_em.token_e != 1:  # merge or split errors
                return False
            last_em = em
        return True

#     @property
#     def type_error(self):
#         return self.__type_error

#     @type_error.setter
#     def type_error(self, error):
#         if error in self.type_errors + [None]:
#             self.__type_error = error
#         else:
#             raise ValueError("Error type assigned '%s' is not one of these: %s" % (error, ', '.join(self.type_errors)))

#     @property
#     def span_error(self):
#         return self.__span_error

#     @span_error.setter
#     def span_error(self, error):
#         if error in self.all_span_errors + [None]:
#             self.__span_error = error
#         else:
#             raise ValueError("Error type assigned '%s' is not one of these: %s" % (error, ', '.join(self.all_span_errors)))

#     def dual_mention_span_relation(self) -> str:
#         """
#         dual_mention_span_relation(predict_mention, gold_mention)
#         returns string (contain | inside | left_extend | right_extend | left_crossed | right_crossed)
#         """
#         pass


class ParserNERErrors:
    """
    Parse NER Errors from a parser
    """
    GOLD_SOURCE_ALIAS = 'gold'
    PREDICT_SOURCE_ALIAS = 'predict'

    def __init__(self, parser: ConllParser, gold_src=None, predict_src=None):
        self.parser = parser

        if gold_src is None:
            self.gold_src = ParserNERErrors.GOLD_SOURCE_ALIAS

        if predict_src is None:
            self.predict_src = ParserNERErrors.PREDICT_SOURCE_ALIAS

        for doc in parser.docs:
            for sentence in doc.sentences:
                ParserNERErrors.set_errors_in_sentence(sentence, self.gold_src, self.predict_src)

    @staticmethod
    def set_errors_in_sentence(sentence: Sentence, gold_src, predict_src) -> None:

        sentence.ems_pairs: Union[EntityMentionsPairs, None] = ParserNERErrors.get_pairs(sentence, gold_src, predict_src)
        sentence.ner_results: List[Union[NERError, NERCorrect]] = None if sentence.ems_pairs is None else \
            sentence.ems_pairs.results

        sentence.set_corrects_from_pairs(sentence.ems_pairs)
        sentence.set_errors_from_pairs(sentence.ems_pairs)
        # TODO: unify the setter or property usage

    @staticmethod
    def get_pairs(sentence: Sentence, gold_src: str, predict_src: str, debug=False) \
            -> Union[None, EntityMentionsPairs]:
        """
        >>> sid, did = 3, 50
        >>> taiwan = ConllToken('Taiwan', 0, sid, did \
        , ners={'gold':ConllNERTag('I-ORG'), 'predict':ConllNERTag('I-LOC')})
        >>> semi = ConllToken('Semiconductor', 1, sid, did,\
        ners={'gold':ConllNERTag('I-ORG'), 'predict':ConllNERTag('I-MISC')})
        >>> manu = ConllToken('Manufacturer', 2, sid, did,\
        ners={'gold':ConllNERTag('I-ORG'), 'predict':ConllNERTag('I-ORG')})
        >>> sen = Sentence([taiwan, semi, manu])
        >>> for token in [taiwan, semi, manu]: \
                token.set_sentence(sen)
        >>> ConllParser.set_entity_mentions_for_one_sentence(sen, ['predict', 'gold'])
        >>> pair = ParserNERErrors.get_pairs(sen, 'gold', 'predict')  # FIXME
        >>> pair.results  #doctest: +ELLIPSIS
        [<error.NERError object at ...
        >>> pair.results[0].type  # FIXME
        {'false_error': None, 'type_error': 'ORG|MISC|LOC->ORG', 'span_error': 'Span Split - 3'}
        """

        if not sentence.entity_mentions_dict[gold_src] and not sentence.entity_mentions_dict[predict_src]:
            return None

        gems = sentence.entity_mentions_dict[gold_src]
        pems = sentence.entity_mentions_dict[predict_src]

        occupied = []  # a list of vacancy. And each vacancy is stored with a list of occupant mention id if occupied

        for idx, em in enumerate(gems + pems):
            # extend the vacancy to be occupied if the vacancy is not enough
            if len(occupied) <= em.token_e:
                for i in range(em.token_e - len(occupied) + 1):
                    occupied.append([])

            if debug:
                print('-------- em.token_b: %i, em.token_e: %i' % (em.token_b, em.token_e))
                print("extended occupied: ", occupied)

            # occupying the vacancy with 'p'(predict) or 'g'(gold) and mention id
            pre = 'g' if idx < len(gems) else 'p'

            for occ in occupied[em.token_b:em.token_e + 1]:
                occ.append((pre, em))

        if debug: print("after occupied: ", occupied)

        pairs = []  # a list of ems_pair. And every ems_pair is a list of entity mention.
        for pos in occupied:
            if debug: print('----- pos: ', pos)
            if pos:

                in_pair_ids = []

                for em in pos:
                    if debug: print("-- em_id: ", em)

                    # return which ems_pair (id) the current mention is in (in_pair_ids)
                    # or return None if there is no ems_pair
                    for idx, pair in enumerate(pairs):
                        if em in pair:
                            in_pair_ids.append(idx)
                        else:
                            in_pair_ids.append(None)

                if debug: print("in_pair_ids: ", in_pair_ids)

                # return the pair_id if there is at least one not None value, else returns None
                in_pair_id = None
                for ioid in in_pair_ids:
                    if ioid is not None:
                        in_pair_id = ioid
                if debug: print("in_pair_id: ", in_pair_id)

                # if there is an existing ems_pair pairing with the current mention: merge
                if in_pair_id is not None:
                    if debug: print("Before update - pairs: ", pairs)
                    pairs[in_pair_id].update(pos)
                    if debug: print("After update - pairs: ", pairs)
                else:  # if not, create a new one and append to pairs
                    if debug: print("Before append - pairs: ", pairs)
                    pairs.append(set(tuple(pos)))
                    if debug: print("After append - pairs: ", pairs)

        if debug: print("pairs: ", pairs)

        # re-sort set in pairs (a list of set) which lost its order during set() based on token_b  # todo: modularize
        sorted_pairs = []
        for pair in pairs:
            sorted_pairs.append(sorted(pair, key=lambda t:t[1].token_b))  # set becomes a list
        pairs = sorted_pairs

        # split to gold and predict entity mentions and create a list of EntityMentionsPair
        ems_pairs: List[EntityMentionsPair] = []
        id_incs = id_incrementer(), id_incrementer()

        for i, pair in enumerate(pairs):  # create a EntityMentionsPair
            pair_p, pair_g = [], []
            for em in pair:  # split to gold mentions and predict mentions (ems_pair is a set, the element is a tuple)
                if em[0] == 'p':
                    pair_p.append(em[1])
                elif em[0] == 'g':
                    pair_g.append(em[1])

            if debug and (not len(pair_p) or not len(pair_g)):
                print('---document %s - %s' % ((list(pair)[0][1].sentence.document.fullid,
                                                list(pair)[0][1].sentence.document)))
                print('--sentence %s - %s' % (list(pair)[0][1].sentence.fullid, list(pair)[0][1].sentence))
                print('-entity_mentions: ', [list(pair)[0][1].print() for pair in pairs])
                print('-pair_p: ', [str(em) for em in pair_p])
                print('-pair_g: ', pair_g)
                print('-ems_pair: ', pair)
                print('-pairs: ', pairs)
                print('-occupied:', occupied)
                print('-occupied text:', [(i, occ[0], str(occ[1])) for i, pos in enumerate(occupied) for occ in pos])
                input()

            # Handle False Negative (FN) and False Positive (FP)
            source_p, type_p, ids_p, source_g, type_g, ids_g = (None, None, None, None, None, None)
            if not pair_p and pair_g:  # FN
                # use the parent_ids of "the other source" since they are in the same sentence
                source_p, type_p, ids_p = predict_src, '', sentence.ids
            elif pair_p and not pair_g:  # FP
                source_g, type_g, ids_g = gold_src, '', sentence.ids
            elif pair_p and pair_g:  # both pair_p and pair_g are not empty
                # just pass None's for they will automatically be extracted
                pass
            else:  # empty
                raise ValueError('Not any mentions in the pair!!!!')

            ems_pairs.append(EntityMentionsPair(i, EntityMentions(pair_g, source_g, type_g, ids_g)
                                                  , EntityMentions(pair_p, source_p, type_p, ids_p), id_incs))

        if debug: print("---------ems_pairs: ", ems_pairs)

        return EntityMentionsPairs(pairs=ems_pairs)


if __name__ == '__main__':
    import doctest

    doctest.run_docstring_examples(NERError, globs=globals())

    train_parser = ConllParser('rcv1.train.compare2')

    train_parser.obtain_statistics(entity_stat=True, source='predict')

    train_parser.set_entity_mentions()

    ParserNERErrors(train_parser)
