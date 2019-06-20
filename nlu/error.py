from typing import Tuple

from nlu.data import *
from nlu.parser import ConllParser


class EntityMentionOverlap(TextList):
    """An EntityMentionOverlap has a gold EntityMentions and a predicted EntityMentions
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
    >>> ol = EntityMentionOverlap(0, tsmcs, ts, error_id_inc)
    >>> print(ol)
    Taiwan Taiwan Semiconductor Manufacturer
    >>> print(ol.fullid)
    D50-S3-OL0
    >>> ol.compare()
    >>> print(ol.result)
    R Diminished None ORG->LOC
    >>> print(str(ol.result.type))
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
                             "mentions in the overlap group")

        self.source = self.mentions[0].source

        ids = copy(self.mentions.ids)
        ids.update({'OL': id_})

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
        return EntityMentionOverlap(self.id, self.gems + other.gems, self.pems + other.pems, self.id_incs)

    def compare(self):
        self.result: Union[NERError, NERCorrect] = NERAnalyzer.extract(self, self.id_incs)
        # self.correct: NERCorrect = None  # todo


class EntityMentionOverlaps(TextList):
    """One sentence will have one EntityMentionOverlaps"""
    def __init__(self, overlaps: List[EntityMentionOverlap]):
        self.overlaps = overlaps
        self.results = [overlap.result for overlap in overlaps]

        try:
            ids = self.overlaps[0].ids
        except IndexError:
            raise IndexError("""Can't access the first element of the overlaps. Overlaps should be empty.
            repr(overlaps): {}
            """.format(repr(self.overlaps)))

        TextList.__init__(self, ids, self.overlaps)


class NERComparison(MD_IDs):

    entity_types = ('PER', 'LOC', 'ORG', 'MISC')

    def __init__(self, em_ol: EntityMentionOverlap, ids):

        self.gems = em_ol.gems
        self.pems = em_ol.pems
        self.gems_total = len(self.gems)
        self.pems_total = len(self.pems)
        self.ems_total = self.gems_total + self.pems_total
        MD_IDs.__init__(self, ids)

    def __str__(self):
        return '\n[predict] ' + str(self.pems) + '\n[gold] ' + str(self.gems) + '\n[type]' + str(self.type)
        # self.type - use type of NERError and NERCorrect


class NERCorrect(NERComparison):

    def __init__(self, em_ol: EntityMentionOverlap, type: str, correct_id):
        ids = copy(em_ol.parent_ids)
        ids.update({'NERCorrect': correct_id})
        super().__init__(em_ol, ids)
        self.type = type


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

    def __init__(self, em_ol: EntityMentionOverlap, type: Dict[str, Union[str, int]], error_id):
        ids = copy(em_ol.parent_ids)
        ids.update({'NERErr': error_id})
        super().__init__(em_ol, ids)
        self.type = type
        self.false_error = self.type['false_error']
        self.span_error = self.type['span_error']
        self.type_error = self.type['type_error']
        self.type = type

    # @property
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
    >>> ol = EntityMentionOverlap(0, tsmcs, ts, id_incs)
    >>> NERAnalyzer.extract(ol, id_incs).type
    {'false_error': None, 'type_error': 'ORG->LOC', 'span_error': 'R Diminished'}
    >>> co_pem = EntityMention([co], source='predict', id_=3)  # test NERCorrect
    >>> co_gem = EntityMention([co], source='gold', id_=3)
    >>> ol_correct = EntityMentionOverlap(1, EntityMentions([co_pem]), EntityMentions([co_gem]), id_incs)
    >>> repr(NERAnalyzer.extract(ol_correct, id_incs))  # doctest: +ELLIPSIS
    '<error.NERCorrect object at ...
    >>> NERAnalyzer.extract(ol_correct, id_incs).type
    'ORG'
    """

    @staticmethod
    def extract(em_ol: EntityMentionOverlap, id_incs: Tuple[id_incrementer, id_incrementer]) \
            -> Union[NERError, NERCorrect]:
        false_error = None
        type_error = None
        span_error = None
        pems: EntityMentions = em_ol.pems
        gems: EntityMentions = em_ol.gems
        ems_total = len(gems) + len(pems)

        error_id, correct_id = id_incs

        if NERAnalyzer.is_mentions_correct(pems, gems):
            return NERCorrect(em_ol, em_ol.gems.type, correct_id)
        else:
            if ems_total == 1:  # False Positive / False Negative
                false_error = 'False Negative ' + gems[0].type if len(gems) == 1 else 'False Positive ' + pems[0].type

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
                    type_error = gems[0].type + '->' + pems[0].type

            elif ems_total >= 3:

                if NERAnalyzer.is_concatenated(pems) and NERAnalyzer.is_concatenated(gems):

                    if NERAnalyzer.has_same_range(pems, gems):
                        if len(pems) == 1:
                            span_error = 'Spans Merged - ' + str(len(gems))
                        elif len(gems) == 1:
                            span_error = 'Span Split - ' + str(len(pems))

                    pts = {pem.type for pem in pems}
                    gts = {gem.type for gem in gems}

                    if not NERAnalyzer.has_same_type(pems, gems):
                        type_error = '|'.join(pts) + '->' + '|'.join(gts)
                    # input('Merge/Split: {}->{}'.format(self.predict_text, self.gold_text))

                else:
                    span_error = 'Complicated - {}->{}'.format(len(gems), len(pems))
                    print('Complicated Case:', [(pem.token_b, pem.token_e, pem.type) for pem in pems],
                          [(gem.token_b, gem.token_e, gem.type) for gem in gems],
                          NERAnalyzer.is_mentions_correct(gems, pems))

        return NERError(em_ol, {'false_error': false_error, 'type_error': type_error,
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

        sentence.em_overlaps: Union[EntityMentionOverlaps, None] = ParserNERErrors.get_overlaps(sentence, gold_src, predict_src)
        sentence.ner_results: List[Union[NERError, NERCorrect]] = None if sentence.em_overlaps is None else \
            sentence.em_overlaps.results

    @staticmethod
    def get_overlaps(sentence: Sentence, gold_src: str, predict_src: str, debug=False) \
            -> Union[None, EntityMentionOverlaps]:
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
        >>> ol = ParserNERErrors.get_overlaps(sen, 'gold', 'predict')  # FIXME
        >>> ol.results  #doctest: +ELLIPSIS
        [<error.NERError object at ...
        >>> ol.results[0].type  # FIXME
        {'false_error': None, 'type_error': 'ORG|MISC|LOC->ORG', 'span_error': 'Span Split - 3'}
        """

        if not sentence.entity_mentions_dict[gold_src] and not sentence.entity_mentions_dict[predict_src]:
            return None

        gems = EntityMentions(sentence.entity_mentions_dict[gold_src])
        pems = EntityMentions(sentence.entity_mentions_dict[predict_src])

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

        overlaps = []  # a list of overlap. And every overlap is a list of entity mention.
        for pos in occupied:
            if debug: print('----- pos: ', pos)
            if pos:

                in_overlap_ids = []

                for em in pos:
                    if debug: print("-- em_id: ", em)

                    # return which overlap (id) the current mention is in (in_overlap_ids)
                    # or return None if there is no overlap
                    for idx, overlap in enumerate(overlaps):
                        if em in overlap:
                            in_overlap_ids.append(idx)
                        else:
                            in_overlap_ids.append(None)

                if debug: print("in_overlap_ids: ", in_overlap_ids)

                # return the overlap_id if there is at least one not None value, else returns None
                in_overlap_id = None
                for ioid in in_overlap_ids:
                    if ioid is not None:
                        in_overlap_id = ioid
                if debug: print("in_overlap_id: ", in_overlap_id)

                # if there is an existing overlap overlapping with the current mention: merge
                if in_overlap_id is not None:
                    if debug: print("Before update - overlaps: ", overlaps)
                    overlaps[in_overlap_id].update(pos)
                    if debug: print("After update - overlaps: ", overlaps)
                else:  # if not, create a new one and append to overlaps
                    if debug: print("Before append - overlaps: ", overlaps)
                    overlaps.append(set(tuple(pos)))
                    if debug: print("After append - overlaps: ", overlaps)

        if debug: print("overlaps: ", overlaps)

        # re-sort set in overlaps (a list of set) which lost its order during set() based on token_b  # todo: modularize
        sorted_overlaps = []
        for overlap in overlaps:
            sorted_overlaps.append(sorted(overlap, key=lambda t:t[1].token_b))  # set becomes a list
        overlaps = sorted_overlaps

        # split to gold and predict entity mentions and create a list of EntityMentionOverlap
        em_overlaps: List[EntityMentionOverlap] = []
        id_incs = id_incrementer(), id_incrementer()

        for i, overlap in enumerate(overlaps):  # create a EntityMentionOverlap
            ol_p, ol_g = [], []
            for em in overlap:  # split to gold mentions and predict mentions (overlap is a set, the element is a tuple)
                if em[0] == 'p':
                    ol_p.append(em[1])
                elif em[0] == 'g':
                    ol_g.append(em[1])

            if debug and (not len(ol_p) or not len(ol_g)):
                print('---document %s - %s' % ((list(overlap)[0][1].sentence.document.fullid,
                                                list(overlap)[0][1].sentence.document)))
                print('--sentence %s - %s' % (list(overlap)[0][1].sentence.fullid, list(overlap)[0][1].sentence))
                print('-entity_mentions: ', [list(overlap)[0][1].print() for overlap in overlaps])
                print('-ol_p: ', [str(em) for em in ol_p])
                print('-ol_g: ', ol_g)
                print('-overlap: ', overlap)
                print('-overlaps: ', overlaps)
                print('-occupied:', occupied)
                print('-occupied text:', [(i, occ[0], str(occ[1])) for i, pos in enumerate(occupied) for occ in pos])
                input()

            # Handle False Negative (FN) and False Positive (FP)
            source_p, type_p, ids_p, source_g, type_g, ids_g = (None, None, None, None, None, None)
            if not ol_p and ol_g:  # FN
                # use the parent_ids of "the other source" since they are in the same sentence
                source_p, type_p, ids_p = predict_src, '', sentence.ids
            elif ol_p and not ol_g:  # FP
                source_g, type_g, ids_g = gold_src, '', sentence.ids
            elif ol_p and ol_g:  # both ol_p and ol_g are not empty
                # just pass None's for they will automatically be extracted
                pass
            else:  # empty
                raise ValueError('Not any mentions in the overlap!!!!')

            em_overlaps.append(EntityMentionOverlap(i, EntityMentions(ol_g, source_g, type_g, ids_g)
                                                    , EntityMentions(ol_p, source_p, type_p, ids_p), id_incs))

        if debug: print("---------em_overlaps: ", em_overlaps)

        return EntityMentionOverlaps(overlaps=em_overlaps)


if __name__ == '__main__':
    import doctest

    doctest.run_docstring_examples(NERError, globs=globals())

    train_parser = ConllParser('rcv1.train.compare2')

    train_parser.obtain_statistics(entity_stat=True, source='predict')

    train_parser.set_entity_mentions()

    ParserNERErrors(train_parser)
