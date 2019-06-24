from typing import Tuple

from nlu.data import *
from nlu.error_structure import EntityMentionsPair, EntityMentionsPairs, NERCorrect, NERErrorComposite
from nlu.parser import ConllParser
from nlu.utils import id_incrementer


class NERErrorExtractor:
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
    >>> NERErrorExtractor.extract(pair, id_incs).type
    {'false_error': None, 'type_error': 'ORG -> LOC', 'span_error': 'R Diminished'}
    >>> co_pem = EntityMention([co], source='predict', id_=3)  # test NERCorrect
    >>> co_gem = EntityMention([co], source='gold', id_=3)
    >>> pair_correct = EntityMentionsPair(1, EntityMentions([co_pem]), EntityMentions([co_gem]), id_incs)
    >>> repr(NERErrorExtractor.extract(pair_correct, id_incs))  # doctest: +ELLIPSIS
    '<...NERCorrect object at ...
    >>> NERErrorExtractor.extract(pair_correct, id_incs).type
    'ORG'
    """

    @staticmethod
    def extract(ems_pair: EntityMentionsPair, id_incs: Tuple[id_incrementer, id_incrementer]) \
            -> Union[NERErrorComposite, NERCorrect]:
        """
        :param ems_pair:
        :param id_incs:
        :return:
        >>> from nlu.data import ConllToken, ConllNERTag, Sentence, EntityMention
        >>> from nlu.error import NERErrorAnnotator, NERErrorExtractor
        >>> import inspect
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
        >>> res = NERErrorExtractor.extract(pair, NERErrorAnnotator.id_incs)
        >>> print(repr(res.__class__), repr(NERErrorComposite))
        <class 'nlu.error_structure.NERErrorComposite'> <class 'nlu.error_structure.NERErrorComposite'>
        >>> isinstance(res, NERErrorComposite)
        True
        >>> pair.set_result(res)
        >>> print(str(pair.result.type))
        {'false_error': None, 'type_error': 'ORG -> LOC', 'span_error': 'R Diminished'}
        """

        pems: EntityMentions = ems_pair.pems
        gems: EntityMentions = ems_pair.gems
        ems_total = len(gems) + len(pems)

        error_id, correct_id = id_incs

        if NERErrorExtractor.is_mentions_correct(pems, gems):
            return NERCorrect(ems_pair, ems_pair.gems.type, correct_id)
        else:
            false_error, span_error, type_error = NERErrorExtractor.get_error(gems, pems)
            return NERErrorComposite(ems_pair, {'false_error': false_error, 'type_error': type_error,
                                  'span_error': span_error}, error_id)

    @staticmethod
    def get_error(gems, pems):
        ems_total = len(gems) + len(pems)
        false_error, span_error, type_error = None, None, None

        if ems_total == 1:  # False Positive / False Negative
            false_error = NERErrorExtractor.get_false_error_eq_one(gems, pems)

        elif ems_total == 2:  # Span Error w/, w/o Type Errors
            span_error = NERErrorExtractor.get_span_error_from_eq_two(gems, pems)
            type_error = NERErrorExtractor.get_type_error_from_eq_two(gems, pems)

        elif ems_total >= 3:

            if NERErrorExtractor.is_concatenated(pems) and NERErrorExtractor.is_concatenated(gems):
                # Merge Or Split
                span_error = NERErrorExtractor.get_merge_or_split_from_ge_three(gems, pems)

                type_error = NERErrorExtractor.get_type_error_from_ge_three(gems, pems)

            else:  # Complicated Case
                span_error = NERErrorExtractor.get_span_error_from_ge_three(gems, pems)
        return false_error, span_error, type_error

    @staticmethod
    def get_type_error_from_ge_three(gems, pems):
        if not NERErrorExtractor.has_same_type(pems, gems):
            return '|'.join(gems.types) + ' -> ' + '|'.join(pems.types)
        else:
            return None

    @staticmethod
    def get_type_error_from_eq_two(gems, pems):
        pt, gt = pems[0].type, gems[0].type
        if pt != gt:  # Type Errors
            return gems[0].type + ' -> ' + pems[0].type
        else:
            return None

    @staticmethod
    def get_span_error_from_ge_three(gems, pems) -> str:
        span_error = 'Complicated - {}->{}'.format(len(gems), len(pems))
        print('Complicated Case:', [(pem.token_b, pem.token_e, pem.type) for pem in pems],
              [(gem.token_b, gem.token_e, gem.type) for gem in gems],
              NERErrorExtractor.is_mentions_correct(gems, pems))
        return span_error

    @staticmethod
    def get_merge_or_split_from_ge_three(gems, pems) -> str:
        if NERErrorExtractor.has_same_range(pems, gems):
            if len(pems) == 1:
                span_error = 'Spans Merged - ' + str(len(gems))
            elif len(gems) == 1:
                span_error = 'Span Split - ' + str(len(pems))
        return span_error

    @staticmethod
    def get_span_error_from_eq_two(gems: EntityMentions, pems: EntityMentions) -> str:
        pb, pe, gb, ge = pems[0].token_b, pems[0].token_e, gems[0].token_b, gems[0].token_e
        if pb == gb and pe != ge:
            span_error = 'R Expansion' if pe > ge else 'R Diminished'
        elif pb != gb and pe == ge:
            span_error = 'L Expansion' if pb < gb else 'L Diminished'
        elif pb != gb and pe != ge:
            if pb < gb:
                span_error = 'L Crossed' if pe < ge else 'RL Expansion'
            else:  #  pb > gb
                span_error = 'R Crossed' if pe > ge else 'RL Diminished'
        else:  # pb == gb and pe == ge
            span_error = None
        return span_error

    @staticmethod
    def get_false_error_eq_one(gems: EntityMentions, pems: EntityMentions) -> str:
        false_error = 'False Negative - ' + gems[0].type if len(gems) == 1 else 'False Positive - ' + pems[0].type
        return false_error

    @staticmethod
    def has_same_type(ems1: EntityMentions, ems2: EntityMentions) -> bool:

        pts = {em1.type for em1 in ems1}
        gts = {em2.type for em2 in ems2}

        return pts == gts

    @staticmethod
    def has_same_range(ems1: EntityMentions, ems2: EntityMentions) -> bool:
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
            if not NERErrorExtractor.is_mention_correct(pem, gem):
                return False
        return True

    @staticmethod
    def is_mention_correct(gold_em: EntityMention, predict_em: EntityMention) -> bool:
        return (predict_em.token_b, predict_em.token_e, predict_em.type) == \
               (gold_em.token_b, gold_em.token_e, gold_em.type)

    @staticmethod
    def is_concatenated(mentions: EntityMentions):
        last_em = None
        for i, em in enumerate(mentions):
            if last_em is not None and em.token_b - last_em.token_e != 1:  # merge or split errors
                return False
            last_em = em
        return True


class MentionsPairsExtractor:
    """Extract EntityMentionsPairs from a Sentence object
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
    >>> ConllParser.set_entity_mentions_for_one_sentence(sen, ['gold', 'predict'])
    >>> pairs = MentionsPairsExtractor.get_pairs(sen, 'gold', 'predict')
    >>> pairs.results  #doctest: +ELLIPSIS
    [<...NERErrorComposite object at ...
    >>> pairs.results[0].type
    {'false_error': None, 'type_error': 'ORG -> LOC|MISC|ORG', 'span_error': 'Span Split - 3'}
    """

    @staticmethod
    def get_pairs(sentence: Sentence, gold_src: str, predict_src: str, debug=False) \
            -> Union[None, EntityMentionsPairs]:

        if not sentence.entity_mentions_dict[gold_src] and not sentence.entity_mentions_dict[predict_src]:
            return None

        gems = sentence.entity_mentions_dict[gold_src]
        pems = sentence.entity_mentions_dict[predict_src]

        occupied = MentionsPairsExtractor.to_occupied(debug, gems, pems)

        pairs = MentionsPairsExtractor.to_sets_of_ems(debug, occupied)

        pairs = MentionsPairsExtractor.sort_sets_of_ems(pairs)

        ems_pairs = MentionsPairsExtractor.to_ems_pairs(debug, gold_src, pairs, predict_src, sentence)

        return EntityMentionsPairs(pairs=ems_pairs)

    @staticmethod
    def to_ems_pairs(debug, gold_src, pairs, predict_src, sentence):
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
                raise ValueError('No any mentions in the pair!!!!')

            ems_pairs.append(EntityMentionsPair(i, EntityMentions(pair_g, source_g, type_g, ids_g)
                                                , EntityMentions(pair_p, source_p, type_p, ids_p), id_incs))
        if debug: print("---------ems_pairs: ", ems_pairs)
        return ems_pairs

    @staticmethod
    def sort_sets_of_ems(pairs):
        # re-sort set in pairs (a list of set) which lost its order during set() based on token_b
        sorted_pairs = []
        for pair in pairs:
            sorted_pairs.append(sorted(pair, key=lambda t: t[1].token_b))  # set becomes a list
        pairs = sorted_pairs
        return pairs

    @staticmethod
    def to_sets_of_ems(debug, occupied):
        pairs = []  # a list of sets, where each set stores an entity mention and its 'p' or 'g' label
        for ems_in_slot in occupied:
            if debug:
                print('----- ems_in_slot: ', ems_in_slot)
            if ems_in_slot:
                pair_ids_with_ems = MentionsPairsExtractor.get_pair_ids_with_ems(ems_in_slot, pairs)

                pair_id = MentionsPairsExtractor.filtered_id(pair_ids_with_ems)

                if debug:
                    print("After update - pairs: ", pairs)

                MentionsPairsExtractor.merge_or_append_ems_to_pairs(ems_in_slot, pair_id, pairs)

                if debug:
                    print("After append - pairs: ", pairs)

        if debug:
            print("pairs: ", pairs)
        return pairs

    @staticmethod
    def merge_or_append_ems_to_pairs(ems_in_slot, pair_id, pairs):
        """if there is an existing ems_pair including the current mention: merge"""
        if pair_id is not None:
            pairs[pair_id].update(ems_in_slot)
        else:  # if not, create a new one and append to pairs
            pairs.append(set(tuple(ems_in_slot)))

    @staticmethod
    def filtered_id(pair_ids_with_ems):
        """return the pair_id if there is at least one not None value, else returns None"""
        pair_id = None
        for id_ in pair_ids_with_ems:
            if id_ is not None:
                pair_id = id_
        return pair_id

    @staticmethod
    def get_pair_ids_with_ems(ems_in_slot, pairs):
        pair_ids_with_ems = []
        for em in ems_in_slot:
            # return which ems_pair (id) the current mention is in (pair_ids_with_ems)
            # or return None if there is no ems_pair
            for id_, pair in enumerate(pairs):
                if em in pair:
                    pair_ids_with_ems.append(id_)
                else:
                    pair_ids_with_ems.append(None)
        return pair_ids_with_ems

    @staticmethod
    def to_occupied(debug, gems, pems):
        occupied = []  # a list of vacancy. And each vacancy stores a list of mention occupant
        for idx, em in enumerate(gems + pems):
            # extend the vacancy to be occupied if the vacancy is not enough
            if len(occupied) <= em.token_e:
                for i in range(em.token_e - len(occupied) + 1):
                    occupied.append([])
            if debug:
                print("extended occupied: ", occupied)

            # occupying the vacancy with 'p'(predict) or 'g'(gold) and mention id
            pre = 'g' if idx < len(gems) else 'p'

            for occ in occupied[em.token_b:em.token_e + 1]:
                occ.append((pre, em))
        if debug:
            print("after occupied: ", occupied)
        return occupied


class NERErrorAnnotator:
    """
    Annotate NER Errors from a parser
    """
    GOLD_SOURCE_ALIAS = 'gold'
    PREDICT_SOURCE_ALIAS = 'predict'
    id_incs = id_incrementer(), id_incrementer()

    @staticmethod
    def annotate(parser, gold_src: str = None, predict_src: str = None):
        gold_src = NERErrorAnnotator.GOLD_SOURCE_ALIAS if gold_src is None else gold_src
        predict_src = NERErrorAnnotator.PREDICT_SOURCE_ALIAS if predict_src is None else predict_src

        for doc in parser.docs:
            NERErrorAnnotator.set_results_in_document(doc, gold_src, predict_src)

    @staticmethod
    def set_results_in_document(doc, gold_src, predict_src):
        for sentence in doc.sentences:
            NERErrorAnnotator.set_results_in_sentence(sentence, gold_src, predict_src)

    @staticmethod
    def set_results_in_sentence(sentence: Sentence, gold_src, predict_src) -> None:

        NERErrorAnnotator.set_ems_pairs_in_sentence(sentence, gold_src, predict_src)
        if sentence.ems_pairs is None:
            sentence.ner_results: List[Union[NERErrorComposite, NERCorrect]] = None
        else:
            for ems_pair in sentence.ems_pairs:
                NERErrorAnnotator.set_result_in_ems_pair(ems_pair)
            NERErrorAnnotator.set_results_in_ems_pairs(sentence.ems_pairs)
            sentence.ner_results = sentence.ems_pairs.results

        sentence.set_corrects_from_pairs(sentence.ems_pairs)
        sentence.set_errors_from_pairs(sentence.ems_pairs)

    @staticmethod
    def set_ems_pairs_in_sentence(sentence: Sentence, gold_src,predict_src):
        sentence.ems_pairs: Union[EntityMentionsPairs, None] = MentionsPairsExtractor.get_pairs(sentence, gold_src,
                                                                                                predict_src)
    @staticmethod
    def set_result_in_ems_pair(ems_pair: EntityMentionsPair):
        ems_pair.set_result(NERErrorExtractor.extract(ems_pair, NERErrorAnnotator.id_incs))  #FIXME: ID

    @staticmethod
    def set_results_in_ems_pairs(ems_pairs: EntityMentionsPairs):
        ems_pairs.results = [pair.result for pair in ems_pairs]
        ems_pairs.corrects = [pair.correct for pair in ems_pairs if pair.correct is not None]
        ems_pairs.errors = [pair.error for pair in ems_pairs if pair.error is not None]


if __name__ == '__main__':
    import doctest

    doctest.testmod()
    # doctest.run_docstring_examples(NERErrorComposite, globs=globals())

    train_parser = ConllParser('../rcv1.train.compare2')

    train_parser.obtain_statistics(entity_stat=True, source='predict')

    train_parser.set_entity_mentions()

    NERErrorAnnotator(train_parser)
