import unittest
from nlu.data import *

import sys
from contextlib import contextmanager
from io import StringIO
import re

@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        

class BaseTestCase(unittest.TestCase):
    
    def assertEqualEllipsis(self, first, second, ellipsis_marker='...', msg=None):
        """    
        Example :
            >>> self.assertEqualEllipsis('foo123bar', 'foo...bar')
        """
        if ellipsis_marker not in second:
            return first == second

        if (
            re.match(
                re.escape(second).replace(re.escape(ellipsis_marker), '(.*?)'),
                first,
                re.M | re.S
            ) is None
        ):
            self.assertMultiLineEqual(
                first,
                second,
                msg
            )
    
    
class TestData(BaseTestCase):
    
    def setUp(self):
        sid, did = 3, 6
        self.tok_tsmc_org = ConllToken('TSMC', 1, sid, did,
                                       ners={'gold': ConllNERTag('I-ORG'), 'predict': ConllNERTag('I-ORG')})
        self.tok_tsmc_org_o = ConllToken('TSMC', 1, sid, did,
                                         ners={'gold': ConllNERTag('O'), 'predict': ConllNERTag('I-ORG')})
        taiwan = ConllToken('Taiwan', 0, sid, did, ners={'gold': ConllNERTag('I-ORG'), \
                                                         'predict': ConllNERTag('I-ORG')})
        semi = ConllToken('Semiconductor', 1, sid, did, ners={'gold': ConllNERTag('I-ORG'), \
                                                              'predict': ConllNERTag('I-ORG')})
        manu = ConllToken('Manufacturer', 2, sid, did, ners={'gold': ConllNERTag('I-ORG'), \
                                                             'predict': ConllNERTag('B-ORG')})
        co = ConllToken('Cooperation', 3, sid, did, ners={'gold': ConllNERTag('I-ORG'), \
                                                          'predict': ConllNERTag('I-ORG')})
        
        tsmc_split = [taiwan, semi, manu, co]
        
        self.tokens = tsmc_split
        
        self.sen = Sentence(tsmc_split)
        
        self.doc = Document([self.sen])

        self.em = EntityMention(tsmc_split, id_=33, source='predict')
        
    
    def testObjectList(self):
        
        class CandyJar(ObjectList):
            def __init__(self, candies):
                ObjectList.__init__(self, candies)
            
        class AGoodCandy(Base):
            def __init__(self, type_):
                self.type = type_
                self.type_ = type_
        
        myfavorite = [AGoodCandy('skittle'), AGoodCandy('chocolate'), AGoodCandy('taffy')]
        
        myjar = CandyJar(myfavorite)
        
        # test slicing __getitem__
        self.assertEqualEllipsis(myjar[1:3], "[Candy(type_='chocolate'), Candy(type_='taffy')]")
        
        # test __iter__
        self.assertEqual([candy.type for candy in myjar], ['skittle', 'chocolate', 'taffy'])
        
        # test __add__
        self.assertEqual(len(myjar + myjar), 6)
        
        # test hashable
        self.assertIsInstance(myjar, Hashable)

        # test references in members and collection
        self.assertIs(myjar.a_good_candys, myfavorite)
        for candy in myfavorite:
            self.assertIs(candy.candy_jar, myjar)
        
        
    def testConllToken(self):
        
        self.assertEqual(self.tok_tsmc_org.fullid, 'D6-S3-T1')

        
    def testSentence(self):
        
        self.assertEqual(len(self.sen), 4)
        com_sents = self.sen + self.sen
        self.assertEqual(repr(com_sents[0]),  # test __add__ and __getitem__
        "ConllToken(text='Taiwan', id_=0, sid=3, did=6, ners={'gold': 'I-ORG', 'predict': 'I-ORG'})")
        
        self.assertEqual(str(self.sen), 'Taiwan Semiconductor Manufacturer Cooperation')
        self.assertEqual((self.sen.fullid, self.sen.ids), ('D6-S3', OrderedDict([('D', 6), ('S', 3)])))
        
        # test
        with captured_output() as (out, err):
            self.sen.print(True)
        lines = out.getvalue().splitlines()
        self.assertEqual(lines[0], 'Taiwan Semiconductor Manufacturer Cooperation(D6-S3)')
        
        self.maxDiff = None  # remove the limitation of the length of the diff message
        self.assertEqualEllipsis(repr(self.sen), "Sentence(tokens=[ConllToken(text='Taiwan', id_=0, sid=3, did=6...), ConllToken(text='Semiconductor', id_=1, sid=3, did=6...), ConllToken(text='Manufacturer', id_=2, sid=3, did=6...), ConllToken(text='Cooperation', id_=3, sid=3, did=6...)]")
        
        # test references and backrefs
        for token1, token2 in zip(self.sen.conll_tokens, self.tokens):
            self.assertEqual(token1, token2)
            
        for token in self.sen:
            self.assertIs(token.sentence, com_sents)  
            # Note on __add__(): the token added will reference to the com_sents rather than self.sen
       
    def testDocument(self):
        
        self.assertEqual(len(self.doc), 1)
        
        self.assertEqual((self.doc.id, self.doc.fullid, self.doc.parent_ids), (6, 'D6', None))
        
        comb_doc = self.doc + self.doc
        self.assertEqual(len(comb_doc), 2)  # test __add__  
                
        self.assertIsInstance(comb_doc[1], Sentence)  
        
        self.assertEqual(comb_doc[1], comb_doc[0])  # test __getitem__
        
    def testEntityMention(self):
        
        self.assertEqual(self.em.fullid, 'D6-S3-EM33')
        
        # hashable test1 (__hash__)
        self.assertIsInstance(self.em, Hashable)
        
        # hashable test2 (__hash__) 
        # (for checking if two mentions are "the same", use set() trick)
        s = set([self.em, self.em])  
        self.assertEqual(len(s), 1)
        
        # test __add__ # test Not consecutive ner positions
        with self.assertRaisesRegex(ValueError, "Not consecutive ner positions"):  
            self.em + self.em
        
        # test ann_str()
        self.assertEqual(self.em.ann_str(), '\x1b[34m[\x1b[0mTaiwan Semiconductor Manufacturer Cooperation\x1b[34m]\x1b[0m\x1b[34mORG\x1b[0m')
        
        # test ann_in_sentence()
        self.assertEqual(self.em.ann_in_sentence(), '\x1b[34m[\x1b[0mTaiwan Semiconductor Manufacturer Cooperation\x1b[34m]\x1b[0m\x1b[34mORG\x1b[0m')
        
#     def testEntityMentions(self):

#         token = ConllToken('TSMC', ners={'predict': ConllNERTag('I-MISC')}, id_=2, sid=6, did=2)
#         >>> token2 = ConllToken('Foundation', ners={'predict': ConllNERTag('I-MISC')}, id_=3, sid=6, did=2)
#         >>> sentence = Sentence([token])
#         >>> token.set_sentence(sentence)
#         >>> token2.set_sentence(sentence)
#         >>> x = EntityMention([token], id_=2, source='predict')
#         >>> y = EntityMention([token2], id_=3, source='predict')
#         >>> pems = EntityMentions([x, y])
#         >>> pems
#         EntityMentions(mentions=[EntityMention(tokens=[ConllToken(text='TSMC', id_=2, sid=6, did=2, ners={'predict': 'I-MISC'})], id_=2, source='predict'), EntityMention(tokens=[ConllToken(text='Foundation', id_=3, sid=6, did=2, ners={'predict': 'I-MISC'})], id_=3, source='predict')], source='predict', ids=OrderedDict([('D', 2), ('S', 6)]))
#         >>> pems.parent_ids
#         OrderedDict([('D', 2)])
#         >>> [ str(token) for pem in pems for token in pem.tokens]  # test __iter__()
#         ['TSMC', 'Foundation']
#         >>> EntityMentions.sep_str(pems, sep='|')
#         'TSMC|Foundation'
#         >>> str(pems)
#         'TSMC Foundation'
#         >>> len((pems + pems)[:])  # test __add__ and __getitem__
#         4
#         >>> EntityMentions([], x.source, x.type, x.parent_ids) \
#          # None EntityMentions test (For False Negative/Positive) # doctest: +ELLIPSIS
#         EntityMentions(mentions=[], source='predict', type_='MISC', ids=OrderedDict([('D', 2), ('S', 6)]))
        
#         self.em
        
        
if __name__ == '__main__':
    unittest.main()
    