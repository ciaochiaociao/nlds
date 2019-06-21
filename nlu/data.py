from collections import OrderedDict, Hashable
from copy import copy
from functools import reduce
from typing import List, Dict, Union

from ansi.color import fg

from nlu.utils import list_to_str, overrides


# from __future__ import annotations
# only work for python 3.7+ but will be implemented in python 4.0. Use literal like 'Text' instead

# todo: get_document|sentence|token_from_fullid|id()
# todo: change all to id_incrementer if possible


class ObjectList:
    """
    Inherited by TextList
    >>> a = ObjectList([1,2,3])
    >>> a[1:3]  # test slicing __getitem__
    [2, 3]
    >>> [i for i in a]  # test __iter__
    [1, 2, 3]
    >>> repr(a + a)  # doctest: +ELLIPSIS
    '<...ObjectList object at ...'
    >>> isinstance(a, Hashable)  # Hashable test
    True
    """
    SEPARATOR = '|'

    def __init__(self, members):
        self.members: List = members

    @overrides(list)
    def __len__(self):
        return len(self.members)

    def separate_str(self):
        return TextList.SEPARATOR.join([str(member) for member in self.members])

    def __hash__(self):
        return hash(repr(self))

    def __getitem__(self, item):
        return self.members[item]

    def __iter__(self):
        return iter(self.members)

    # this should be overridden in the child's __add__. The following just shows an example.
    def __add__(self, other):
        return ObjectList(self.members + other.members)


class MD_IDs:
    def __init__(self, ids):
        """
        >>> ids = MD_IDs(ids=OrderedDict({'D':2, 'S': 1, 'T': 7}))
        >>> ids.fullid
        'D2-S1-T7'
        >>> ids.parent_fullid
        'D2-S1'
        >>> MD_IDs(ids=OrderedDict({'D':5, 'S':3, 'EM': 2})).fullid
        'D5-S3-EM2'
        >>> MD_IDs(ids=OrderedDict({'D':5, 'S':None, 'T': 2})).fullid # None in ids
        Traceback (most recent call last):
        ...
        ValueError: There is at least one id not assigned in ids: OrderedDict([('D', 5), ('S', None), ('T', 2)])
        >>> MD_IDs().fullid
        Traceback (most recent call last):
        ...
        TypeError: __init__() missing 1 required positional argument: 'ids'
        """
        self.ids: OrderedDict = ids

        # parse ids
        reversed_ids = reversed(ids)

        self.id = self.ids[next(reversed_ids)]
        self.fullid: str = MD_IDs.concat_ids(self.ids)
        self.parent_ids = MD_IDs.get_parent_ids(self.ids)
        self.parent_fullid = MD_IDs.concat_ids(self.parent_ids)

    @property
    def ids(self) -> OrderedDict:
        return self.__ids

    @ids.setter
    def ids(self, ids: OrderedDict):
        if None in ids.values():
            raise ValueError(f'There is at least one id not assigned in ids: {ids}')
        else:
            self.__ids = ids

    @staticmethod
    def concat_ids(ids: 'Union[None, OrderedDict[str, int]]') -> str:
        """
        >>> MD_IDs.concat_ids(OrderedDict({'D':1, 'S':2}))
        'D1-S2'
        >>> MD_IDs.concat_ids(OrderedDict({'A':3, 'B':4, 'CC':5, 'DE':6}))
        'A3-B4-CC5-DE6'
        >>> none1, none2, none3 = None, OrderedDict(), OrderedDict({'A':3, 'B':4, 'CC':None, 'DE':6})
        >>> MD_IDs.concat_ids(none1), MD_IDs.concat_ids(none2), MD_IDs.concat_ids(none3)
        ('', '', '')
        """
        if not ids or type(ids) is not OrderedDict or None in ids.values():
            return ''
        return '-'.join([pre + str(id_) for pre, id_ in ids.items()])

    @staticmethod
    def get_parent_ids(ids: OrderedDict) -> Union[OrderedDict, None]:
        """
        >>> MD_IDs.get_parent_ids(OrderedDict({'D': 2, 'S': 1, 'T': 7}))
        OrderedDict([('D', 2), ('S', 1)])
        >>> MD_IDs.get_parent_ids(OrderedDict({'D': 2}))
        """
        return OrderedDict(list(ids.items())[:-1]) or None

    def __eq__(self, other):
        return self.ids == other.ids

    def __hash__(self):
        # make children class hashable. This trick is especially needed when __eq__ is overridden
        return hash(repr(self))


class TextWithIDs(MD_IDs):
    """abstract class: directly inherited by TextList, Token
    >>> class SomeText(TextWithIDs):
    ...     def __init__(self, text, ids):
    ...         self.text = text
    ...         super().__init__(ids)
    ...     def __str__(self):
    ...         return self.text
    >>> SomeText('A sample text', ids=OrderedDict({'D':2, 'S': 4})).print(True)
    A sample text(D2-S4)
    """

    def __init__(self, ids):
        self.text: str = str(self)
        MD_IDs.__init__(self, ids)

    def print(self, detail=False):
        if detail:
            print('%s(%s)' % (self.text, self.fullid))
        else:
            print('%s' % self.text)

    # abstract method
    def __str__(self):
        raise NotImplementedError


class TextList(ObjectList, TextWithIDs):
    """Directly inherited by Sentence, Document, EntityMention, ...
    >>> sid, did = 3, 6
    >>> taiwan = ConllToken('Taiwan', 0, sid, 111, ners={'gold':ConllNERTag('I-ORG'), 'predict':ConllNERTag('I-ORG')})
    >>> semi = ConllToken('Semiconductor', 1, sid, did, ners={'gold':ConllNERTag('I-ORG'), \
    'predict':ConllNERTag('I-ORG')})
    >>> manu = ConllToken('Manufacturer', 2, 999, did, ners={'gold':ConllNERTag('I-ORG'), \
    'predict':ConllNERTag('B-ORG')})
    >>> co = ConllToken('Cooperation', 3, sid, did, ners={'gold':ConllNERTag('B-ORG'), 'predict':ConllNERTag('I-ORG')})
    >>> t = TextList(OrderedDict({'D': did, 'S': sid, 'EM': 4}), [semi, co])
    >>> t.fullid  # test MD_IDs
    'D6-S3-EM4'
    >>> TextList(OrderedDict({'D': did, 'S': sid, 'EM': 5}), [manu, co])
    Traceback (most recent call last):
    ...
    ValueError: The members "Manufacturer" of class "ConllToken" in "LIST[ConllToken]" should have same parent ids
                    full ids of the members: [OrderedDict([('D', 6), ('S', 999)]), OrderedDict([('D', 6), ('S', 3)])]
    >>> TextList(OrderedDict({'D': 255, 'S': sid, 'EM': 5}), [manu, co])
    Traceback (most recent call last):
    ...
    ValueError: The members "Manufacturer" of class "ConllToken" in "LIST[ConllToken]" should have same parent ids
                    full ids of the members: [OrderedDict([('D', 6), ('S', 999)]), OrderedDict([('D', 6), ('S', 3)])]
    >>> TextList(OrderedDict({'D': did, 'S': None, 'EM': 5}), [manu, co])
    Traceback (most recent call last):
    ...
    ValueError: The members "Manufacturer" of class "ConllToken" in "LIST[ConllToken]" should have same parent ids
                    full ids of the members: [OrderedDict([('D', 6), ('S', 999)]), OrderedDict([('D', 6), ('S', 3)])]
    >>> [ str(men) for men in t ]
    ['Semiconductor', 'Cooperation']
    >>> repr(t+t)  # doctest:+ELLIPSIS
    '<...TextList object at ...>'
    >>> TextList(OrderedDict({'D': did, 'S': sid, 'EM': 5}), [])  # doctest:+ELLIPSIS
    <...TextList object at ...>
    """
    SEPARATOR = '|'

    def __init__(self, ids, members):
        # sanity check
        mem_p_ids = [member.parent_ids for member in members]
        # self_p_ids = [IDs.get_parent_ids(ids)]
        if mem_p_ids and not reduce(lambda mb1, mb2: mb1 if mb1 == mb2 else False, mem_p_ids):
            raise ValueError(
                """The members "{0}" of class "{1}" in "LIST[{1}]" should have same parent ids
                full ids of the members: {2}"""
                .format(members[0], members[0].__class__.__name__, mem_p_ids))

        ObjectList.__init__(self, members)
        # a hack: TextWithIDs requires its __str__ to be overridden, while the __str__ is defined in their another
        # inherited parent class ObjectList. To have the str
        # The order of the inheritance should be TextList(ObjectList, MD_IDs)
        # and the call of their __init__ should be ObjectList.__init__() followed by TextWithIDs.__init__()

        TextWithIDs.__init__(self, ids)

    @overrides(TextWithIDs)
    def __str__(self):
        return self.sep_str()

    def sep_str(self, sep=' '):  # TODO: abstracted to a public utility function
        return sep.join([str(member) for member in self.members]) if self.members else ''

    @overrides(ObjectList)
    def __add__(self, other):
        return TextList(self.ids, self.members + other.members)


class Tag:
    def __init__(self, type_):
        self.type: str = type_

    def __repr__(self):
        return self.type


class ConllNERTag(Tag):  # TODO: Create an EntityTag and an ConllEntityTag class
    __types = ['I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'B-PER', 'B-ORG', 'B-LOC', 'B-MISC', 'O']

    @property
    def type(self):
        return self.__type

    @type.setter
    def type(self, type_):
        if type_ in self.__types:
            self.__type = type_
            if len(type_.split('-')) == 2:
                self.prefix, self.suffix = type_.split('-')
            else:
                self.prefix, self.suffix = None, None
        else:
            raise ValueError("Tag assigned '%s' is not one of these: %s" % (type_, ', '.join(self.__types)))


class ConllChunkTag(Tag):  # todo
    pass


class ConllPosTag(Tag):  # todo
    pass


class Token(TextWithIDs):
    def __init__(self, text, id_, sid, did):
        self.sid = sid
        self.did = did
        self.sentence = None
        ids = OrderedDict({'D': self.did, 'S': self.sid, 'T': id_})
        self.text = text
        TextWithIDs.__init__(self, ids)

    def set_sentence(self, sentence: 'Sentence'):
        self.sentence = sentence

    @overrides(TextWithIDs)
    def __str__(self):
        return self.text

    # TODO: reference to document


class ConllToken(Token):
    def __init__(self, text, id_, sid, did, poss=None, chunks=None, ners=None):
        """
        >>> ConllToken('TSMC', id_=1, sid=2, did=3, ners={'predict': ConllNERTag('I-MISC')}).fullid
        'D3-S2-T1'
        """
        self.poss: Dict[str, str] = poss
        self.chunks: Dict[str, str] = chunks
        self.ners: Dict[str, ConllNERTag] = ners
        super().__init__(text, id_, sid, did)


class Sentence(TextList):
    def __init__(self, tokens: List[Token]):
        """
        >>> sid, did = 3, 6
        >>> taiwan = ConllToken('Taiwan', 0, sid, did, ners={'gold':ConllNERTag('I-ORG'), \
        'predict':ConllNERTag('I-ORG')})
        >>> semi = ConllToken('Semiconductor', 1, sid, did, ners={'gold':ConllNERTag('I-ORG'), \
        'predict':ConllNERTag('I-ORG')})
        >>> manu = ConllToken('Manufacturer', 2, sid, did, ners={'gold':ConllNERTag('I-ORG'), \
        'predict':ConllNERTag('B-ORG')})
        >>> co = ConllToken('Cooperation', 3, sid, did, ners={'gold':ConllNERTag('B-ORG'), \
        'predict':ConllNERTag('I-ORG')})
        >>> sen = Sentence([taiwan, semi, manu, co])
        >>> sen.print(True)
        Taiwan Semiconductor Manufacturer Cooperation(D6-S3)
        >>> doc = Document([sen])
        >>> sen.set_document(doc)
        >>> print(sen.document)
        Taiwan Semiconductor Manufacturer Cooperation
        >>> len(sen)
        4
        >>> sen.fullid, sen.ids
        ('D6-S3', OrderedDict([('D', 6), ('S', 3)]))
        >>> str(sen)  # test __str__
        'Taiwan Semiconductor Manufacturer Cooperation'
        >>> repr((sen+sen)[0])  # test __add__ and __getitem__  # doctest: +ELLIPSIS
        '<...ConllToken object at ...>'
        """
        self.tokens: List[Token] = tokens
        self.entity_mentions_dict: Dict[str, List[EntityMention]] = {'predict': [], 'gold': []}  # todo: change to EntityMentions
        self.id: int = tokens[0].sid
        self.did: int = tokens[0].did
        self.document = None
        ids = self.tokens[0].parent_ids
        TextList.__init__(self, ids, tokens)

    def set_document(self, document: 'Document') -> None:
        self.document = document

    def set_errors_from_pairs(self, pairs) -> None:
        if pairs is None:
            self.errors = None
        else:
            self.errors = pairs.errors

    def set_corrects_from_pairs(self, pairs) -> None:
        if pairs is None:
            self.corrects = None
        else:
            self.corrects = pairs.corrects

    def set_pairs(self, pairs: List) -> None:
        self.pairs = pairs

    def print_corrects(self) -> None:
        if self.corrects:
            for correct in self.corrects:
                print(str(correct))

    def print_errors(self) -> None:
        if self.errors:
            for error in self.errors:
                print(str(error))

    @overrides(TextList)
    def __add__(self, other):
        return Sentence(self.members + other.members)


class Document(TextList):
    """
    >>> sid, did = 3, 6
    >>> taiwan = ConllToken('Taiwan', 0, sid, did, ners={'gold':ConllNERTag('I-ORG'), \
    'predict':ConllNERTag('I-ORG')})
    >>> semi = ConllToken('Semiconductor', 1, sid, did, ners={'gold':ConllNERTag('I-ORG'), \
    'predict':ConllNERTag('I-ORG')})
    >>> manu = ConllToken('Manufacturer', 2, sid, did, ners={'gold':ConllNERTag('I-ORG'), \
    'predict':ConllNERTag('B-ORG')})
    >>> co = ConllToken('Cooperation', 3, sid, did, ners={'gold':ConllNERTag('B-ORG'), \
    'predict':ConllNERTag('I-ORG')})
    >>> sen = Sentence([taiwan, semi, manu, co])
    >>> sen.print(True)
    Taiwan Semiconductor Manufacturer Cooperation(D6-S3)
    >>> doc = Document([sen])
    >>> len(doc)
    1
    >>> doc.id, doc.fullid, doc.parent_ids
    (6, 'D6', None)
    >>> repr(doc + doc)  # test __add__  # doctest: +ELLIPSIS
    '<...Document object at ...>'
    >>> (doc + doc)[0]  # test __add__ and __getitem__  # doctest: +ELLIPSIS
    <...Sentence object at ...
    """
    def __init__(self, sentences: List[Sentence]):

        self.id = sentences[0].did
        self.sentences: List[Sentence] = sentences
        ids = self.sentences[0].parent_ids
        TextList.__init__(self, ids, sentences)

    @overrides(TextList)
    def __add__(self, other):
        return Document(self.members + other.members)


class EntityMention(TextList):
    """
    >>> token1 = ConllToken('Welcome', ners={'predict': ConllNERTag('I-MISC')}, id_=0, sid=6, did=2)
    >>> token2 = ConllToken('TSMC', ners={'predict': ConllNERTag('I-MISC')}, id_=1, sid=6, did=2)
    >>> token3 = ConllToken('!', ners={'predict': ConllNERTag('I-MISC')}, id_=2, sid=6, did=2)
    >>> sentence = Sentence([token1, token2, token3])
    >>> for token in [token1, token2, token3]: token.set_sentence(sentence)
    >>> x = EntityMention([token2], id_=33, source='predict')
    >>> x.fullid
    'D2-S6-EM33'
    >>> isinstance(x, Hashable)  # hashable test1
    True
    >>> s = set([x, x])  #hashable test2 (for checking if two mentions are "the same" using set() trick)
    >>> (x+x)[:]  # test __add__ and __getitem__  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Not consecutive ner positions ...
    >>> print(x.ann_str())  # blue
    \x1b[34m[\x1b[0mTSMC\x1b[34m]\x1b[0m\x1b[34mMISC\x1b[0m
    >>> x.source = 'gold'; print(x.ann_str())  # yellow
    \x1b[33m[\x1b[0mTSMC\x1b[33m]\x1b[0m\x1b[33mMISC\x1b[0m
    >>> print(x.ann_in_sentence())  # yellow
    Welcome\x1b[33m[\x1b[0mTSMC\x1b[33m]\x1b[0m\x1b[33mMISC\x1b[0m!
    """

    def __init__(self, tokens, id_, source):
        # sanity check
        if len(set([token.ners[source].suffix for token in tokens])) != 1:
            raise ValueError("""Not consistent ner types in tokens of the entity mention {} : {}
            sentence: {}
            """.format(id_
                       , [(token.fullid, token.text, token.ners[source].type) for token in tokens]
                       , tokens[0].sentence))
        elif [token.id for token in tokens] != list(range(tokens[0].id, tokens[-1].id + 1)):
            raise ValueError('Not consecutive ner positions (id) in tokens of the entity mention {}: id - {}'
                             .format(id_, [token.id for token in tokens]))

        self.tokens: List[Token] = tokens
        self.source = source

        self.sid, self.did = tokens[0].sid, tokens[0].did
        self.type: str = tokens[0].ners[source].suffix
        self.token_b = tokens[0].id
        self.token_e = tokens[-1].id

        self.sentence = tokens[0].sentence
        self.document = self.sentence.document

        self.ems_pair = None

        ids = copy(self.tokens[0].parent_ids)
        ids.update({'EM': id_})

        TextList.__init__(self, ids, tokens)

    @overrides(TextList)
    def print(self, detail=False):
        if detail:
            print('(%s-%s)%s - %s' % (self.source, self.fullid, self.type, self.text))
        else:
            print('(%s)%s - %s' % (self.source, self.type, self.text))

    def ann_in_sentence(self) -> str:
        return list_to_str(self.sentence[:self.token_b]) + self.ann_str() + list_to_str(self.sentence[self.token_e+1:])

    @overrides(TextList)
    def __add__(self, other):
        return EntityMention(self.members + other.members, self.id, self.source)

    def ann_str(self) -> str:
        if self.source == 'predict':
            color = fg.blue
        elif self.source == 'gold':
            color = fg.yellow
        else:
            raise ValueError('souce is neither "predict" nor "gold"')
        return color('[') + str(self) + color(']') + color(self.type)


class EntityMentions(TextList):
    """A user-defined list of 'EntityMention's.
    - ids: its ids is the same as the ids of the sentence it resides in
    >>> token = ConllToken('TSMC', ners={'predict': ConllNERTag('I-MISC')}, id_=2, sid=6, did=2)
    >>> token2 = ConllToken('Foundation', ners={'predict': ConllNERTag('I-MISC')}, id_=3, sid=6, did=2)
    >>> sentence = Sentence([token])
    >>> token.set_sentence(sentence)
    >>> token2.set_sentence(sentence)
    >>> x = EntityMention([token], id_=2, source='predict')
    >>> y = EntityMention([token2], id_=3, source='predict')
    >>> pems = EntityMentions([x, y])
    >>> pems.parent_ids
    OrderedDict([('D', 2)])
    >>> [ str(token) for pem in pems for token in pem.tokens]  # test __iter__()
    ['TSMC', 'Foundation']
    >>> EntityMentions.sep_str(pems, sep='|')
    'TSMC|Foundation'
    >>> str(pems)
    'TSMC Foundation'
    >>> len((pems + pems)[:])  # test __add__ and __getitem__
    4
    >>> EntityMentions([], x.source, x.type, x.parent_ids) \
     # None EntityMentions test (For False Negative/Positive) # doctest: +ELLIPSIS
    <...EntityMentions object at ...
    """

    def __init__(self, mentions: List[EntityMention], source: str = None, type_: str = None
                 , ids: OrderedDict = None):

        self.mentions = mentions
        self.all_tokens: List[Token] = [token for mention in mentions for token in mention.tokens]
        try:
            ids = copy(mentions[0].parent_ids)
            self.source = mentions[0].source
            self.type = mentions[0].type  # todo: sanity check
            self.token_b = mentions[0].token_b
            self.token_e = mentions[-1].token_e
            self.token_bs = [mention.token_b for mention in mentions]
            self.token_es = [mention.token_e for mention in mentions]
            self.sentence = mentions[0].sentence
            self.types: List[str] = [mention.type for mention in mentions]
        except IndexError:
            self.source = source
            self.type = type_
            ids = ids

        TextList.__init__(self, ids, mentions)

    @overrides(TextList)
    def __add__(self, other):
        return EntityMentions(self.members + other.members)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
