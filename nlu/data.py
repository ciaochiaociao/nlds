from collections import OrderedDict, Hashable
from copy import copy
from functools import reduce
from typing import List, Dict, Union, Optional

from ansi.color import fg

from nlu.utils import list_to_str, overrides, sep_str, camel_to_snake, setup_logger

import inspect
import re
# from __future__ import annotations
# only work for python 3.7+ but will be implemented in python 4.0. Use literal like 'Text' instead

# todo: get_document|sentence|token_from_fullid|id()
# todo: change all to id_incrementer if possible

dataLogger = setup_logger('data_logger', '../data.log')


def get_func_keys(func):
    
    return inspect.signature(func).parameters.keys()

    
def prepr(obj):
    return obj.prepr()


def default_repr_format(obj, pretty=False):
    kwargs = {k: obj.__getattribute__(k) for k in get_func_keys(obj.__init__)}
    
    if not pretty:
        kwargs_str = ', '.join(['%s=%r' % (k, v) for k, v in kwargs.items() if v is not None])
        return '{self.__class__.__name__}({})'.format(kwargs_str, self=obj)
    else:  # TODO
        kwargs_str = ', '.join(['{}={}'.format(k, prepr(v) if prepr in dir(v) else repr(v)) for k, v in kwargs.items() if v is not None])
        return '{self.__class__.__name__}(\n\t{})'.format(kwargs_str, self=obj)


class Base(object):

    def __repr__(self):
        return default_repr_format(self)
    
    def prepr(self):
        return default_repr_format(self, True)

class ObjectList(Base):
    """
    Inherited by TextList
    """

    SEPARATOR = '|'

    def __init__(self, members):
        
        self_name = camel_to_snake(self.__class__.__name__)
        
        # santiy check
        if len(set([type(member) for member in members])) > 1:
            dataLogger.warning('Not all members in ObjectList are the same: {}'.format(members))
            dataLogger.warning('This first element is set with self.{}'.format(
                mems_attr_name))
        
        # set refs
        try:
            mems_attr_name = camel_to_snake(members[0].__class__.__name__) + 's'  #TODO: better plural 
            self.__setattr__(mems_attr_name, members)  # container.<member_name>  ex-entity_mention.tokens
        except IndexError:
            dataLogger.warning('No members in building a ObjectList')
        except AttributeError:
            dataLogger.warning('Not an object in building a ObjectList')

        self.members: List = members  #container.members  ex-entity_mention.members
        
        # set backrefs
        for member in members:
            if 'parents' not in dir(member):
                member.containers = {}
            else:
                if self not in member.containers.values():
                    member.containers.update({self_name: self})
                    # member.containers[<container_name>] ex-token.containers['entity_mention']
            member.__setattr__(self_name, self)  # member.<container_name>  ex-token.entity_mention

    @overrides(list)
    def __len__(self):
        return len(self.members)

    def sep_str(self, sep=None):
        return sep_str(self.members, sep=sep)

    def __hash__(self):
        return hash(repr(self))

    def __getitem__(self, item):
        return self.members[item]

    def __iter__(self):
        return iter(self.members)

    # this should be overridden in the child's __add__. The following just shows an example.
    def __add__(self, other):  
        """Note on __add__(): the tokens 'added' will reference to the newly created sentence rather than the original through token.sentence"""
        return ObjectList(self.members + other.members)


class MD_IDs(Base):
    def __init__(self, ids):
        """
        >>> ids = MD_IDs(ids=OrderedDict({'D':2, 'S': 1, 'T': 7}))
        >>> ids
        MD_IDs(ids=OrderedDict([('D', 2), ('S', 1), ('T', 7)]))
        >>> ids.fullid
        'D2-S1-T7'
        >>> ids.parent_fullid
        'D2-S1'
        >>> MD_IDs(ids=OrderedDict({'D':5, 'S':3, 'EM': 2})).fullid
        'D5-S3-EM2'
        >>> MD_IDs(ids=OrderedDict({'D':5, 'S':None, 'T': 2})).fullid # None in ids
        Traceback (most recent call last):
        ...
        ValueError: ids is None or there is at least one id not assigned in ids: OrderedDict([('D', 5), ('S', None), ('T', 2)])
        >>> MD_IDs().fullid
        Traceback (most recent call last):
        ...
        TypeError: __init__() missing 1 required positional argument: 'ids'
        """
        self.ids: OrderedDict = ids

        # parse ids
        reversed_ids = reversed(ids)

        self.id = self.ids[next(reversed_ids)]
        self.id_ = self.id
        self.fullid: str = MD_IDs.concat_ids(self.ids)
        self.parent_ids = MD_IDs.get_parent_ids(self.ids)
        self.parent_fullid = MD_IDs.concat_ids(self.parent_ids)

    @property
    def ids(self) -> OrderedDict:
        return self.__ids

    @ids.setter
    def ids(self, ids: OrderedDict):
        if ids is None or None in ids.values():
            raise ValueError(f'ids is None or there is at least one id not assigned in ids: {ids}')
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
    
#     def __repr__(self):
#         """
#         >>> repr(MD_IDs(ids=OrderedDict({'D':5, 'S':3, 'EM': 2})))
#         "MD_IDs(OrderedDict([('D', 5), ('S', 3), ('EM', 2)]))"
#         """
#         return '{self.__class__.__name__}({self.ids!r})'.format(self=self)

    @staticmethod
    def parse_fullid_str(str_) -> List:
        """
        >>> MD_IDs.parse_fullid_str('D9-S3-E2')
        [('D', '9'), ('S', '3'), ('E', '2')]
        """
        a = [re.split('(\d+)', id_wp, maxsplit=1) for id_wp in str_.split('-')]
        d = [tuple([i for i in s if i != '']) for s in a]
        return d

    @classmethod
    def from_str(cls, str_: str) -> 'MD_IDs':
        """
        >>> MD_IDs.from_str('D9-S3-E2')
        MD_IDs(ids=OrderedDict([('D', '9'), ('S', '3'), ('E', '2')]))
        """
        return cls.from_list(MD_IDs.parse_fullid_str(str_))
    
    @classmethod
    def from_list(cls, list_: list) -> 'MD_IDs':
        """
        >>> MD_IDs.from_list([('D', '9'), ('S', '3'), ('E', '2')])
        MD_IDs(ids=OrderedDict([('D', '9'), ('S', '3'), ('E', '2')]))
        """
        return cls(OrderedDict(list_))

class TextWithIDs(MD_IDs):
    """abstract class: directly inherited by TextList, Token
    >>> class SomeText(TextWithIDs):
    ...     def __init__(self, text, ids):
    ...         self.text = text
    ...         super().__init__(ids)
    ...     def __str__(self):
    ...         return self.text
    >>> text_obj = SomeText('A sample text', ids=OrderedDict({'D':2, 'S': 4}))
    >>> text_obj.print(True)
    A sample text(D2-S4)
    >>> repr(text_obj)
    "SomeText(text='A sample text', ids=OrderedDict([('D', 2), ('S', 4)]))"
    """

    def __init__(self, ids):
        try:
            self.text: str = str(self)
        except AttributeError:  # for those who implement their __str__ with the use of other attribute
            pass
            
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
    "TextList(ids=OrderedDict([('D', 6), ('S', 3), ('EM', 4)]), members=[ConllToken(text='Semiconductor', id_=1, sid=3, did=6...), ConllToken(text='Cooperation', id_=3, sid=3, did=6...), ConllToken(text='Semiconductor', id_=1, sid=3, did=6...), ConllToken(text='Cooperation', id_=3, sid=3, did=6...)])"
    >>> TextList(OrderedDict({'D': did, 'S': sid, 'EM': 5}), [])  # doctest:+ELLIPSIS
    TextList(ids=OrderedDict([('D', 6), ('S', 3), ('EM', 5)]), members=[])
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

    @overrides(ObjectList)
    def __add__(self, other):
        return TextList(self.ids, self.members + other.members)

#     def __repr__(self):
#         return '{self.__class__.__name__}({self.ids!r}, {self.members!r})'.format(self=self)


class InSentence:

    def set_sentence(self, sentence: 'Sentence') -> None:
        self.sentence = sentence
    
class InDocument:

    def set_document(self, document: 'Document') -> None:
        self.document = document


class Tag(Base):
    def __init__(self, type_):
        self.type: str = type_
#         self.type_: str = type_

    def __repr__(self):
        return repr(self.type)


class ConllNERTag(Tag):  # TODO: Create an EntityTag and an ConllEntityTag class

    @property
    def type(self):
        return self.__type

    @type.setter
    def type(self, type_):
        self.__type = type_
        if len(type_.split('-', 1)) == 2:
            self.prefix, self.suffix = type_.split('-', 1)
        else:
            self.prefix, self.suffix = None, None

            
class ConllChunkTag(Tag):  # todo
    pass


class ConllPosTag(Tag):  # todo
    pass


class Token(TextWithIDs, InSentence):
    def __init__(self, text, id_, sid, did):
        """
        >>> t = Token('TSMC', 1, 2, 3)
        >>> repr(t)
        "Token(text='TSMC', id_=1, sid=2, did=3)"
        """
        self.sid = sid
        self.did = did
        
        # attribute for backref
        self.sentence = None
        
        ids = OrderedDict({'D': self.did, 'S': self.sid, 'T': id_})
        self.text = text
        TextWithIDs.__init__(self, ids)  # self.id is set here

#     def set_sentence(self, sentence: 'Sentence'):
#         self.sentence = sentence

    @overrides(TextWithIDs)
    def __str__(self):
        return self.text

#     # TODO: reference to document


class ConllToken(Token):
    
    auto_id = -1
    
    def __init__(self, text, id_, sid, did, poss=None, chunks=None, ners=None, conf=None, line=None, line_no=None):
        """
        >>> ct = ConllToken(text='TSMC',id_=1,sid=2,did=3,ners={'predict': ConllNERTag('I-MISC')})
        >>> repr(ct)
        "ConllToken(text='TSMC', id_=1, sid=2, did=3, ners={'predict': 'I-MISC'})"
        >>> ct.fullid
        'D3-S2-T1'
        """
        self.poss: Dict[str, str] = poss
        self.chunks: Dict[str, str] = chunks
        self.ners: Dict[str, Tag] = ners
        self.conf = conf
        self.line = line
        self.line_no = line_no
        super().__init__(text, id_, sid, did)
        
    @classmethod
    def easy_build(cls, text, ner=None, gner=None, restart=False, id_=None, sid=999, did=99, **kwargs):
        """
        >>> ConllToken.easy_build('TSMC', 'I-ORG', restart=True)
        ConllToken(text='TSMC', id_=0, sid=999, did=99, ners={'predict': 'I-ORG'})
        >>> ConllToken.easy_build('Morris', 'I-PER')
        ConllToken(text='Morris', id_=1, sid=999, did=99, ners={'predict': 'I-PER'})
        >>> ConllToken.easy_build('Taipei', 'I-LOC', id_=12)
        ConllToken(text='Taipei', id_=12, sid=999, did=99, ners={'predict': 'I-LOC'})
        """
        if restart:
            cls.auto_id = -1
        
        if id_ is None:
            cls.auto_id += 1
            id_ = cls.auto_id
        
        # just text
        if ner is None and gner is None:
            return cls(text, id_, sid, did, **kwargs)

        # with predict or gold tag
        ners_dict = {}
        if ner is not None:
            ners_dict.update({'predict': ConllNERTag(ner)})
        if gner is not None:
            ners_dict.update({'gold': ConllNERTag(gner)})
            
        return cls(text, id_, sid, did, ners=ners_dict, **kwargs)
        
    @classmethod
    def bulk_easy_build(cls, texts: List[str], pners: List[str]=None, gners=None, **kwargs):
        """
        >>> tokens = ConllToken.bulk_easy_build(['TSMC', 'is', 'in', 'Hsinchu', 'Taiwan', '.'], ['I-ORG', 'O', 'O', 'I-LOC', 'B-LOC', 'O'])
        >>> len(tokens)
        6
        """
        cls.auto_id = -1
        kwargs = {k:v for k, v in kwargs.items() if k != 'id_'}

        assert (gners is None or len(gners) == len(texts)) and \
            (pners is None or len(pners) == len(texts)), 'The number of tokens and ner tags do not match!'
            
        
        argses = zip(*[arg for arg in [texts, pners, gners] if arg is not None])
        return [cls.easy_build(*args, **kwargs) for args in argses]
        
       
    @classmethod
    def bulk_from_str(cls, text_str, pner_str=None, gner_str=None, sep=' ', **kwargs):
        """
        >>> tokens = ConllToken.bulk_from_str('NLU Lab is in Taipei Taiwan directed by Keh Yih Su .', 'I-ORG I-ORG O O I-LOC B-LOC O O I-PER I-PER I-PER O')
        >>> len(tokens)
        12
        """
        
        args = [arg.strip().split(sep) for arg in [text_str, pner_str, gner_str] if arg is not None]
        return cls.bulk_easy_build(*args, **kwargs)

    
class Sentence(TextList, InDocument):
    def __init__(self, tokens: List[Token]):
        """
        >>> tokens = ConllToken.bulk_easy_build(['TSMC', 'is', 'in', 'Hsinchu', 'Taiwan', '.'], ['I-ORG', 'O', 'O', 'I-LOC', 'B-LOC', 'O'])
        >>> sen = Sentence(tokens)
        >>> sen  # doctest: +ELLIPSIS
        Sentence(tokens=[ConllToken..., ConllToken(text='.', id_=5, sid=999, did=99...)])
        >>> sen.print(True)
        TSMC is in Hsinchu Taiwan .(D99-S999)
        >>> len(sen)
        6
        >>> sen.fullid, sen.ids
        ('D99-S999', OrderedDict([('D', 99), ('S', 999)]))
        >>> str(sen)  # test __str__
        'TSMC is in Hsinchu Taiwan .'
        >>> repr((sen+sen)[0])  # test __add__ and __getitem__  # doctest: +ELLIPSIS
        "ConllToken(text='TSMC', id_=0, sid=999, did=99, ners={'predict': 'I-ORG'})"
        >>> doc = Document([sen])
        >>> print(sen.document)
        TSMC is in Hsinchu Taiwan .
        """
        self.tokens: List[Token] = tokens
        self.entity_mentions_dict: Dict[str, List[EntityMention]] = {'predict': [], 'gold': []}  # todo: change to EntityMentions
        self.id: int = tokens[0].sid
        self.did: int = tokens[0].did
        
        # attributes for backrefs
        self.document = None
        self.ems_pairs: Optional['EntityMentionsPairs'] = None
            
            
        self.ner_results: Optional[List['NERComparisonWithID']] = None
        self.ner_corrects: Optional[List['NERCorrect']] = None
        self.ner_errors: Optional[List['NERErrorComposite']] = None
        ids = self.tokens[0].parent_ids
        TextList.__init__(self, ids, tokens)

#     def set_document(self, document: 'Document') -> None:
#         self.document = document

    def set_errors_from_pairs(self, pairs) -> None:  # FIXME: No longer used?
        if pairs is None:
            self.ner_errors = []
        else:
            self.ner_errors = pairs.errors

    def set_corrects_from_pairs(self, pairs) -> None:  # FIXME: No longer used?
        if pairs is None:
            self.ner_corrects = []
        else:
            self.ner_corrects = pairs.corrects

    def print_corrects(self) -> None:
        if self.ner_corrects:
            for correct in self.ner_corrects:
                print(str(correct))

    def ner_print_errors(self) -> None:
        if self.ner_errors:
            for error in self.ner_errors:
                print(str(error))

    @overrides(TextList)
    def __add__(self, other):
        return Sentence(self.members + other.members)

#     def __repr__(self):
#         return '{self.__class__.__name__}({self.tokens!r})'.format(self=self) 
    
    @classmethod
    def easy_build(cls, *args, **kwargs):
        """
        >>> sen = Sentence.easy_build(['NLU', 'Lab', 'is', 'in', 'Taipei', 'Taiwan', 'directed', 'by', 'Keh', 'Yih', 'Su', '.'], ['I-ORG', 'I-ORG', 'O', 'O', 'I-LOC', 'B-LOC', 'O', 'O', 'I-PER', 'I-PER', 'I-PER', 'O'])
        >>> len(sen)
        12
        """
        return cls(ConllToken.bulk_easy_build(*args, **kwargs))
    
    @classmethod
    def from_str(cls, *args, **kwargs):
        """
        >>> Sentence.from_str('NLU Lab is in Taipei Taiwan directed by Keh Yih Su .', 'I-ORG I-ORG O O I-LOC B-LOC O O I-PER I-PER I-PER O', 'I-ORG I-ORG O O I-LOC I-LOC O O O I-PER I-PER O')  # doctest: +ELLIPSIS
        Sentence(...ners={'predict': 'I-ORG', 'gold': 'I-ORG'}...)
        """
        return cls(ConllToken.bulk_from_str(*args, **kwargs))
    
    def set_entity_mentions(self, sources: List=['predict', 'gold']) -> None:
        """chunk entity mentions for all sources (i.e. predict, gold) from `ConllToken`s in a sentence
        effect: set sentence.entity_mentions_dict ({'predict': `EntityMention`s})
        >>> sen = Sentence.from_str('NLU Lab is in Taipei Taiwan directed by Keh Yih Su .', pner_str='I-ORG I-ORG O O I-LOC B-LOC O O I-PER I-PER I-PER O', gner_str='I-ORG I-ORG O O I-LOC I-LOC O O O I-PER I-PER O')
        >>> sen.set_entity_mentions()
        >>> len(sen.gems), len(sen.pems), len(sen.ems)
        (3, 4, 7)
        >>> sen.pems[0]  # doctest: +ELLIPSIS
        EntityMention(...'NLU'...'predict': 'I-ORG'...'Lab'...predict': 'I-ORG'... source='predict')
        >>> len(sen.pems[0]), sen.pems[0].type
        (2, 'ORG')
        """
        for source in sources:

            entity_mentions = self.get_entity_mentions(source)
            
            if entity_mentions:
                self.entity_mentions_dict[source] = entity_mentions
                
                if source == 'predict':
                    self.pems = entity_mentions
                elif source == 'gold':
                    self.gems = entity_mentions
                else:
                    raise ValueError('source should be either "predict" or "gold": {}'.format(source))
        
        self.ems = []
        self.ems += [ em for ems in self.entity_mentions_dict.values() for em in ems]
    
    def get_entity_mentions(self, source: str) -> List['EntityMention']:
        """chunk entity mentions for all sources (i.e. predict, gold) from `ConllToken`s in a sentence
        >>> sen = Sentence.from_str('NLU Lab is in Taipei Taiwan directed by Keh Yih Su .', pner_str='I-ORG I-ORG O O I-LOC B-LOC O O I-PER I-PER I-PER O', gner_str='I-ORG I-ORG O O I-LOC I-LOC O O O I-PER I-PER O')
        >>> gems = sen.get_entity_mentions('gold')
        >>> pems = sen.get_entity_mentions('predict')
        >>> len(gems)
        3
        >>> len(pems)
        4
        >>> pems[0]  # doctest: +ELLIPSIS
        EntityMention(...'NLU'...'predict': 'I-ORG'...'Lab'...predict': 'I-ORG'... source='predict')
        >>> len(pems[0])
        2
        >>> pems[0].type
        'ORG'
        """

        tokens_tray: List[Token] = []
        entity_mentions: List['EntityMention'] = []
#         last_token = ConllToken(text='', ner='O')
        last_token = None
        eid = 0

        for i, token in enumerate(self.tokens):
            # A. Boundary Detected: create an EntityMention from the tokens_tray and append to entity_mentions and
            # empty tokens_tray if the tokens_tray is not empty (the last token is of entity tag)
            # (Boundary: 'O', "B-" prefix, "I-" with different suffix )
            # B. Entity Tags Detected: add Token to tokens_tray
            # (not 'O')

            if token.ners[source].type == 'O' or token.ners[source].prefix == 'B' or not last_token \
                    or token.ners[source].suffix != last_token.ners[source].suffix:  # Boundary detected
                if tokens_tray:
                    entity_mentions.append(EntityMention(tokens_tray, id_=eid, source=source))
                    eid += 1
                    tokens_tray = []

            if token.ners[source].type != 'O':
                tokens_tray.append(token)

            last_token = token

        # at the end of a sentence: 
        #  - add the last entity mention if it exists
        if tokens_tray:
            entity_mentions.append(EntityMention(tokens_tray, id_=eid, source=source))
    
        return entity_mentions
    
    def get_ann_sent(self, ems: 'EntityMentions', color=fg.blue, color_em=True) -> str:
        """highlight all entity mentions with one color in the sentence"""
        return self.get_diff_ann_sent([ems], [color], color_em)
    
    def get_diff_ann_sent(self, all_ems: List['EntityMentions'], colors=None, color_em=True) -> str:
        """highlight different entity mentions with different color in the sentence"""
        def _split(split: list, a: list):

            import copy

            result = copy.copy(list(split))

            if split[0] != 0:
                result.insert(0, 0)
            if split[-1] != len(a):
                result.append(len(a))

            ranges = []
            for i in range(len(result)-1):
                ranges.append((result[i], result[i + 1]))

            return ranges

        def _get_oppo_split(splits: list, a: list):
            fa = sorted({j for i in splits for j in i})
            return sorted(set(_split(fa, a)) - set(splits))

        def get_str(sentence: 'Sentence', range_: tuple, color=fg.blue, color_em=True):
            if color_em:
                return color('[' + list_to_str(sentence[slice(*range_[0])]) + ']' + range_[1].type)
            else:
                return color('[') + list_to_str(sentence[slice(*range_[0])]) + color(']' + range_[1].type)

        def _get_ems_dict_with_range(ems: 'EntityMentions', *args, **kwargs) -> list:
            ems_range = [((em.token_b, em.token_e+1), em) for em in ems]
            ems_dict = [(range_[0][0], get_str(self, range_, *args, **kwargs)) for range_ in ems_range]
            return ems_dict, ems_range
        
        # sanity check
        for ems in all_ems:
            if any(em.sentence is not self for em in ems):
                raise ValueError('Entity mention in sentence.ems are not in the same sentence: {}'.format(self.ems))
        
        if colors is None:
            colors = [fg.blue] * len(all_ems)
        
        all_ems_range = set()
        all_ems_dict = []
        for ems, color in zip(all_ems, colors):
            ems_dict, ems_range = _get_ems_dict_with_range(ems, color)
            all_ems_range |= set(ems_range)
            all_ems_dict.extend(ems_dict)
                        
        all_ems_range = sorted(all_ems_range)
        
        non_ems_range = _get_oppo_split(dict(all_ems_range).keys(), self)
        non_ems_dict = [(range_[0], list_to_str(self[slice(*range_)])) for range_ in non_ems_range]
        
        all_dict = all_ems_dict + non_ems_dict
        
        sentence = list_to_str(dict(sorted(all_dict)).values())
        return sentence
    
    
class Document(TextList):
    """
    >>> sen1 = Sentence.from_str('NLU Lab is in Taipei Taiwan directed by Keh Yih Su .', 'I-ORG I-ORG O O I-LOC B-LOC O O I-PER I-PER I-PER O', 'I-ORG I-ORG O O I-LOC I-LOC O O O I-PER I-PER O')
    >>> sen2 = Sentence.from_str('Natural Language Processing is so fun !', 'I-MISC I-MISC I-MISC O O O O', 'O I-MISC I-MISC O O O O')
    >>> sen1.print(True)
    NLU Lab is in Taipei Taiwan directed by Keh Yih Su .(D99-S999)
    >>> doc = Document([sen1, sen2])
    >>> str(doc)
    'NLU Lab is in Taipei Taiwan directed by Keh Yih Su . Natural Language Processing is so fun !'
    >>> len(doc)
    2
    >>> doc.id, doc.fullid, doc.parent_ids
    (99, 'D99', None)
    >>> len(doc + doc)  # test __add__  # doctest: +ELLIPSIS
    4
    >>> (doc + doc)[0]  # test __add__ and __getitem__  # doctest: +ELLIPSIS
    Sentence(...)
    """
    def __init__(self, sentences: List[Sentence]):

        self.id = sentences[0].did
#         self.sentences: List[Sentence] = sentences
        ids = sentences[0].parent_ids
        TextList.__init__(self, ids, sentences)

    @overrides(TextList)
    def __add__(self, other):
        return Document(self.members + other.members)


class EntityMention(TextList, InSentence):
    """
    >>> sen = Sentence.from_str('NLU Lab is in Taipei Taiwan directed by Keh Yih Su .', 'I-ORG I-ORG O O I-LOC B-LOC O O I-PER I-PER I-PER O', 'I-ORG I-ORG O O I-LOC I-LOC O O O I-PER I-PER O')
    >>> em1 = EntityMention(sen[0:2], id_=33, source='predict')
    >>> em2 = EntityMention(sen[4:6], id_=34, source='gold')
    >>> em1.fullid
    'D99-S999-EM33'
    >>> isinstance(em1, Hashable)  # hashable test1
    True
    >>> s = set([em1, em2])  #hashable test2 (for checking if two mentions are "the same" using set() trick)
    >>> em1+em1  # test __add__ and __getitem__  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Not consecutive ner positions ...
    >>> em1+em2  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Not consistent ner types ...
    >>> print(em1.ann_str())  # blue
    \x1b[34m[\x1b[0mNLU Lab\x1b[34m]\x1b[0m\x1b[34mORG\x1b[0m
    >>> em1.source = 'gold'; print(em1.ann_str())  # yellow
    \x1b[33m[\x1b[0mNLU Lab\x1b[33m]\x1b[0m\x1b[33mORG\x1b[0m
    >>> print(em1.ann_in_sentence())  # yellow
    \x1b[33m[\x1b[0mNLU Lab\x1b[33m]\x1b[0m\x1b[33mORG\x1b[0mis in Taipei Taiwan directed by Keh Yih Su .
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

        # set refs automatically
        try:
            self.sentence = tokens[0].sentence
        except AttributeError:
            pass

        self.ems_pair = None

        ids = copy(self.tokens[0].parent_ids)
        if source == 'predict':
            prefix = 'PEM'
        elif source == 'gold':
            prefix = 'GEM'
        else:
            raise ValueError('source nees to be either "predict" or "gold"')
        ids.update({prefix: id_})

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

#     @overrides(MD_IDs)  # this will make unhashable
#     def __eq__(self):  # TODO
#         MD_IDs.__eq__(self)
        

class EntityMentions(TextList):
    """A user-defined list of 'EntityMention's.
    - ids: its ids is the same as the ids of the sentence it resides in
    >>> sen = Sentence.from_str('NLU Lab is in Taipei Taiwan directed by Keh Yih Su .', 'I-ORG I-ORG O O I-LOC B-LOC O O I-PER I-PER I-PER O', 'I-ORG I-ORG O O I-LOC I-LOC O O O I-PER I-PER O')
    >>> sen.set_entity_mentions()
    >>> pems = EntityMentions(sen.pems)
    >>> len(pems), type(pems[0:3]), type(pems[0]), len(pems[0:3]), type(pems[-1])
    (4, <class '__main__.EntityMentions'>, <class '__main__.EntityMention'>, 3, <class '__main__.EntityMention'>)
    >>> pems.parent_ids
    OrderedDict([('D', 99)])
    >>> {token.fullid:str(token) for pem in pems for token in pem.tokens}  # test __iter__()
    {'D99-S999-T0': 'NLU', 'D99-S999-T1': 'Lab', 'D99-S999-T4': 'Taipei', 'D99-S999-T5': 'Taiwan', 'D99-S999-T8': 'Keh', 'D99-S999-T9': 'Yih', 'D99-S999-T10': 'Su'}
    >>> EntityMentions.sep_str(pems, sep='|')
    'NLU Lab|Taipei|Taiwan|Keh Yih Su'
    >>> str(pems)
    'NLU Lab Taipei Taiwan Keh Yih Su'
    >>> len((pems + pems)[:])  # test __add__ and __getitem__
    8
    >>> EntityMentions([], pems.source, pems.type, pems.parent_ids) \
     # None EntityMentions test (For False Negative/Positive)
    EntityMentions(mentions=[], source='predict', type_='ORG', ids=OrderedDict([('D', 99)]))
    >>> print(pems[1:2].get_diff_ann_sent(pems[:1]+pems[2:], colors=[fg.green, fg.red]))
    \x1b[31m[NLU Lab]ORG\x1b[0m is in \x1b[32m[Taipei]LOC\x1b[0m \x1b[31m[Taiwan]LOC\x1b[0m directed by \x1b[31m[Keh Yih Su]PER\x1b[0m .
    >>> print(pems[1:2].get_ann_em_in_sent())
    \x1b[34m[NLU Lab]ORG\x1b[0m is in \x1b[32m[Taipei]LOC\x1b[0m \x1b[34m[Taiwan]LOC\x1b[0m directed by \x1b[34m[Keh Yih Su]PER\x1b[0m .
    """

    def __init__(self, mentions: List[EntityMention], source: str = None, type_: str = None
                 , ids: OrderedDict = None):

        self.mentions = mentions
        self.all_tokens: List[Token] = [token for mention in mentions for token in mention.tokens]
        try:
            ids = copy(mentions[0].parent_ids)  # FIXME: parent_ids should be D and S instead? 
            # (this occurs because there is no its id for EntityMentions for now) (Add id 'PEMS0' and 'GEMS0')
            self.sid, self.did = ids['S'], ids['D']
            self.source = mentions[0].source
            self.type = mentions[0].type
            self.token_b = mentions[0].token_b
            self.token_e = mentions[-1].token_e
            self.token_bs = [mention.token_b for mention in mentions]
            self.token_es = [mention.token_e for mention in mentions]
            self.sentence = mentions[0].sentence
            self.types: List[str] = [mention.type for mention in mentions]
            self.type_ = type_
        except IndexError:  # no mention
            dataLogger.error(mentions)
            self.source = source
            self.type = type_
            self.type_ = type_
            ids = ids

        TextList.__init__(self, ids, mentions)

    @overrides(TextList)
    def __add__(self, other) -> 'EntityMentions':
        return EntityMentions(self.members + other.members)

    def get_ann_sent(self, *args, **kwargs):
        """porting to sentence.get_ann_sent"""
        return self.sentence.get_ann_sent(self, *args, **kwargs)
    
    def get_diff_ann_sent(self, other_ems, *args, **kwargs):
        """porting to sentence.get_diff_ann_sent"""
        return self.sentence.get_diff_ann_sent([self, other_ems], *args, **kwargs)
    
    def get_ann_em_in_sent(self, colors=[fg.green, fg.blue]):
        """highlight the entity mentions and the others with two different colors in the setnecne"""
        other_ems = EntityMentions([em for em in self.sentence.entity_mentions_dict[self.source] if em not in self])
        return self.get_diff_ann_sent(other_ems, colors)
    
    def __getitem__(self, item) -> 'EntityMentions':
        """
        extend TextList.__getitem__()
        return EntityMentions
        """
        if isinstance(item, slice):
            return EntityMentions(TextList.__getitem__(self, item))
        elif isinstance(item, int):
            return TextList.__getitem__(self, item)
        else:
            raise IndexError("Only integer slice or index are avaiable for EntityMentions")

if __name__ == '__main__':
    import doctest

    failure_count, test_count = doctest.testmod()
    
    if failure_count == 0:
        
        print('{} tests passed!!!!!!'.format(test_count))
