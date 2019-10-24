from typing import Tuple

from nlu.data import *
from nlu.error_structure import EntityMentionsPair, EntityMentionsPairs, NERCorrect, NERErrorComposite, SpanError, \
    MergeSplitError, MentionTypeError, ComplicateError, FalseError
from nlu.parser import ConllParser
from nlu.utils import id_incrementer


class NERErrorExtractor:
    """
    >>> sen = Sentence.from_str('NLU Lab is in Taipei Taiwan directed by Keh Yih Su .', 'I-ORG I-ORG O O I-LOC I-ORG O O I-PER I-PER I-PER O', 'I-ORG I-ORG O O I-LOC I-LOC O O O I-PER I-PER O')
    >>> sen.set_entity_mentions()
    >>> id_incs = id_incrementer(), id_incrementer()
    >>> pairs = MentionsPairsExtractor.get_pairs(sen)
    >>> error_comp = NERErrorExtractor.extract(pairs[1], pairs[1].id_incs)
    >>> type(error_comp)
    <class 'nlu.error_structure.NERErrorComposite'>
    >>> {t: str(o) for t, o in error_comp.type.items()}
    {'false_error': 'None', 'type_error': 'LOC -> LOC|ORG', 'span_error': 'Taipei Taiwan -> Taipei|Taiwan'}
    >>> corr = NERErrorExtractor.extract(pairs[0], pairs[0].id_incs)
    >>> type(corr)
    <class 'nlu.error_structure.NERCorrect'>
    >>> pairs[1].set_result(error_comp)
    >>> {t: str(o) for t, o in pairs[1].result.type.items()}
    {'false_error': 'None', 'type_error': 'LOC -> LOC|ORG', 'span_error': 'Taipei Taiwan -> Taipei|Taiwan'}
    """

    @staticmethod
    def extract(ems_pair: EntityMentionsPair, id_incs: Tuple[id_incrementer, id_incrementer]) \
            -> Union[NERErrorComposite, NERCorrect]:
        
        pems: EntityMentions = ems_pair.pems
        gems: EntityMentions = ems_pair.gems
        ems_total = len(gems) + len(pems)

        error_id, correct_id = id_incs

        if NERErrorExtractor.is_mentions_correct(pems, gems):
            return NERCorrect(ems_pair, ems_pair.gems.type, correct_id)
        else:
            false_error, span_error, type_error = NERErrorExtractor.get_error(ems_pair)
            return NERErrorComposite(ems_pair, {'false_error': false_error, 'type_error': type_error,
                                  'span_error': span_error}, error_id)

    @staticmethod
    def get_error(ems_pair: EntityMentionsPair):
        gems = ems_pair.gems
        pems = ems_pair.pems
        ems_total = len(gems) + len(pems)
        false_error, span_error, type_error = None, None, None

        if ems_total == 1:  # False Positive / False Negative
            false_error = NERErrorExtractor.get_false_error_eq_one(ems_pair)

        elif ems_total == 2:  # Span Error w/, w/o Type Errors
            span_error = NERErrorExtractor.get_span_error_from_eq_two(ems_pair)
            type_error = NERErrorExtractor.get_type_error_from_eq_two(ems_pair)

        elif ems_total >= 3:

            if NERErrorExtractor.is_concatenated(pems) and NERErrorExtractor.is_concatenated(gems) and \
                    NERErrorExtractor.has_same_range(pems, gems) and (len(pems) == 1 or len(gems) == 1):
                # Merge Or Split
                span_error = NERErrorExtractor.get_merge_or_split_from_ge_three(ems_pair)

                type_error = NERErrorExtractor.get_type_error_from_ge_three(ems_pair)

            else:  # Complicated Case  
                """Complicated Case: 
                    - [O][OO] <-> [OO][O] both pems and gems have the same range +
                    but none of them have only one entity mention
                    - [OO]O[O] <-> [OOOO] not concat
                    - [OO][O] <-> [O][OOO] not same range
                """
                span_error = NERErrorExtractor.get_span_error_from_ge_three(ems_pair)

                type_error = NERErrorExtractor.get_type_error_from_ge_three(ems_pair)
        return false_error, span_error, type_error

    @staticmethod
    def get_type_error_from_ge_three(ems_pair: EntityMentionsPair) -> Optional[MentionTypeError]:
        pems, gems = ems_pair.pems, ems_pair.gems
        if not NERErrorExtractor.has_same_type(pems, gems):
            return MentionTypeError(ems_pair)
        else:
            return None

    @staticmethod
    def get_type_error_from_eq_two(ems_pair: EntityMentionsPair) -> Optional[MentionTypeError]:
        pt, gt = ems_pair.pems[0].type, ems_pair.gems[0].type
        if pt != gt:  # Type Errors
            return MentionTypeError(ems_pair)
        else:
            return None

    @staticmethod
    def get_span_error_from_ge_three(ems_pair: EntityMentionsPair) -> str:
        pems, gems = ems_pair.pems, ems_pair.gems
        span_error = ComplicateError(ems_pair)
        return span_error

    @staticmethod
    def get_merge_or_split_from_ge_three(ems_pair: EntityMentionsPair) -> MergeSplitError:
        pems, gems = ems_pair.pems, ems_pair.gems
        if len(pems) == 1:  # ex - P: [OOO] G: [O][OO]
            span_error = MergeSplitError(ems_pair, 'Spans Merged')
        elif len(gems) == 1:  # ex - P: [O][OO] G: [OOO]
            span_error = MergeSplitError(ems_pair, 'Span Split')
        return span_error

    @staticmethod
    def get_span_error_from_eq_two(ems_pair: EntityMentionsPair) -> Optional[SpanError]:

        pb, pe = ems_pair.pems[0].token_b, ems_pair.pems[0].token_e
        gb, ge = ems_pair.gems[0].token_b, ems_pair.gems[0].token_e

        if pb == gb and pe != ge:
            direction, type_ = ('Right', 'Expansion') if pe > ge else ('Right', 'Diminished')
        elif pb != gb and pe == ge:
            direction, type_ = ('Left', 'Expansion') if pb < gb else ('Left', 'Diminished')
        elif pb != gb and pe != ge:
            if pb < gb:
                direction, type_ = ('Left', 'Crossed') if pe < ge else ('Right Left', 'Expansion')
            else:  # pb > gb
                direction, type_ = ('Right', 'Crossed') if pe > ge else ('Right Left', 'Diminished')
        else:  # pb == gb and pe == ge
            direction, type_ = None, None

        span_error = SpanError(ems_pair, direction, type_) if (direction, type_) != (None, None) else None
        return span_error

    @staticmethod
    def get_false_error_eq_one(ems_pair: EntityMentionsPair) -> FalseError:

        false_error = FalseError(ems_pair, 'False Negative') if len(ems_pair.gems) == 1 else \
            FalseError(ems_pair, 'False Positive')

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
    >>> sen = Sentence.from_str('NLU Lab is in Taipei Taiwan directed by Keh Yih Su .', 'I-ORG I-ORG O O I-LOC B-LOC O O I-PER I-PER I-PER O', 'I-ORG I-ORG O O I-LOC I-LOC O O O I-PER I-PER O')
    >>> sen.set_entity_mentions()
    >>> pairs = MentionsPairsExtractor.get_pairs(sen)
    >>> isinstance(pairs[0], EntityMentionsPair)
    True
    >>> print(str(pairs[0]))
    D99-S999-PAIR0: (G) \x1b[33m[NLU Lab]ORG\x1b[0m is in Taipei Taiwan directed by Keh Yih Su . (P) \x1b[34m[NLU Lab]ORG\x1b[0m is in Taipei Taiwan directed by Keh Yih Su .
    """
    
#     >>> pairs.results  #doctest: +ELLIPSIS
#     [<...NERErrorComposite object at ...
#     >>> pairs.results[0].type
#     {'false_error': None, 'type_error': 'ORG -> LOC|MISC|ORG', 'span_error': 'Span Split - 3'}

    @staticmethod
    def get_pairs(sentence: Sentence, gold_src: str='gold', predict_src: str='predict', debug=False) \
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


class NERErrorAnnotator:  # TODO: takes DocumentsWithEMAnn returns DocumentsWithErrorAnn
    """
    Annotate NER Errors for a parser
    """
    GOLD_SOURCE_ALIAS = 'gold'
    PREDICT_SOURCE_ALIAS = 'predict'
    id_incs = id_incrementer(), id_incrementer()

    @staticmethod
    def annotate(parser, gold_src: str = None, predict_src: str = None):
        """
        >>> parser = ConllParser('../test/testa.pred.gold')
        >>> parser.set_entity_mentions()
        >>> NERErrorAnnotator.annotate(parser)
        >>> parser.error_overall_stats()
        ---Overall Results---
        found entity mentions: 5942
        true entity mentions: 5942
        correct_total:  5941
        error_total:  1
        precision: 99.98%
        recall: 99.98%
        macro-f1: 99.98%
        corrects ratio: 99.98%
        all corrects and errors 5942
        the number of sentences with/without entities (predict + gold): 2605 (80%), 645 (20%)        
        """
        gold_src = NERErrorAnnotator.GOLD_SOURCE_ALIAS if gold_src is None else gold_src
        predict_src = NERErrorAnnotator.PREDICT_SOURCE_ALIAS if predict_src is None else predict_src

        for doc in parser.docs:
            NERErrorAnnotator.set_results_in_document(doc, gold_src, predict_src)

        # set back references for parser
        parser.ner_errors = [error for doc in parser.docs for error in doc.ner_errors]
        parser.ner_corrects = [correct for doc in parser.docs for correct in doc.ner_corrects]
        parser.ner_results = [result for doc in parser.docs for result in doc.ner_results]

    @staticmethod
    def set_results_in_document(doc, gold_src, predict_src):
        for sentence in doc.sentences:
            NERErrorAnnotator.set_results_in_sentence(sentence, gold_src, predict_src)

        # set back ref for doc
        doc.ner_errors = [error for sent in doc for error in sent.ner_errors]
        doc.ner_corrects = [correct for sent in doc for correct in sent.ner_corrects]
        doc.ner_results = [result for sent in doc for result in sent.ner_results]

    @staticmethod
    def set_results_in_sentence(sentence: Sentence, gold_src, predict_src) -> None:
        """
        effect: set sentence.ems_pairs, sentence.ner_results, sentence.corrects, sentence.errors
        """
        NERErrorAnnotator.set_ems_pairs_in_sentence(sentence, gold_src, predict_src)
        if sentence.ems_pairs is None:
            sentence.ner_results: List[Union[NERErrorComposite, NERCorrect]] = []
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
        ems_pair.set_result(NERErrorExtractor.extract(ems_pair, NERErrorAnnotator.id_incs))  # FIXME: ID

    @staticmethod
    def set_results_in_ems_pairs(ems_pairs: EntityMentionsPairs):
        ems_pairs.results = [pair.result for pair in ems_pairs]
        ems_pairs.corrects = [pair.correct for pair in ems_pairs if pair.correct is not None]
        ems_pairs.errors = [pair.error for pair in ems_pairs if pair.error is not None]


if __name__ == '__main__':
    import doctest

    # doctest.run_docstring_examples(NERErrorComposite, globs=globals())
    
    failure_count, test_count = doctest.testmod()
    
    if failure_count == 0:
        
        print('{} tests passed!!!!!!'.format(test_count))
    
#         parser = ConllParser('../test/testa.pred.gold')

#         parser.obtain_statistics(entity_stat=True, source='predict')

#         parser.set_entity_mentions()

#         NERErrorAnnotator(parser)
