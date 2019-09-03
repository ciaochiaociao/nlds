from nlu.data import *


class ConllParser(Base):  #TODO: create methods that returns ConllDocuments
    """Column Parser for CoNLL03 formatted text file"""
    TAGGERSOURCE = 'gold'

    def __init__(self, filepath: str, cols_format: List[Dict[str, Union[str, int]]] = None) -> None:
        """
            :param filepath: The filename. Note that the file loaded should end with two blank lines!!!
            :param cols_format:
                - type: predict | gold
                - col_num: starts from 0
                - tagger: ner | pos | chunk
        >>> import os.path
        >>> cols_format = [{'type': 'predict', 'col_num': 1, 'tagger': 'ner'}, {'type': 'gold', 'col_num': 2, 'tagger': 'ner'}]
        >>> path = os.path.dirname(os.path.abspath(__file__)) + '/../test/train.pred.gold'
        >>> train_parser = ConllParser(path, cols_format)
        >>> train_parser  # doctest: +ELLIPSIS
        ConllParser(filepath='.../test/train.pred.gold', cols_format=[{'type': 'predict', 'col_num': 1, 'tagger': 'ner'}, {'type': 'gold', 'col_num': 2, 'tagger': 'ner'}])
        >>> train_parser.obtain_statistics(entity_stat=True, source='predict')  # doctest: +ELLIPSIS
        ---...
        Document number:  946
        Sentence number:  14041
        Token number:  203621
        PER: 6599 (28%)
        LOC: 7134 (30%)
        ORG: 6324 (27%)
        MISC: 3435 (15%)
        TOTAL: 23492
        >>> train_parser.set_entity_mentions()
        """
        # attributes
        self.docs = []
        self.filepath = filepath
        if cols_format is None:
            self.cols_format = [{'type': 'predict', 'col_num': 1, 'tagger': 'ner'},
                                {'type': 'gold', 'col_num': 2, 'tagger': 'ner'}]
        else:
            self.cols_format = cols_format

        tok_dicts = ConllParser.parse_conll_to_tok_dicts(self.filepath, self.cols_format)
        self.docs = ConllParser.build_md_docs_from_tok_dicts(tok_dicts)
        
        # optional back reference functionality: token1.sentence, sentence5.document
        ConllParser.add_back_refs_for_md_docs(self.docs)

    @staticmethod
    def parse_conll_to_tok_dicts(filepath: str, cols_format) -> List[List[List[dict]]]:
        """read a conll-formatted text file to a hierarchical collection of token dictionaries (tok_dict is like below)
        input:
        filepath - the file path of the conll-formatted file
        cols_format - [{'type': 'predict', 'col_num': 1, 'tagger': 'ner'},
                        {'type': 'gold', 'col_num': 2, 'tagger': 'ner'},
                        {'type': 'predict', 'col_num': 3, 'tagger': 'pos'},
                        ...]
        output: 
        same as the input of function build_md_docs_from_tok_dicts() where
        tok_dict = {
            'text': 'Obama',
            'poss': {'predict': 'NN', 'gold': 'NN'},
            'chunks': {'predict': 'I-NNP', 'gold': 'I-NNP'},
            'ners': {'predict': 'I-PER', 'gold': 'I-PER'}
            }
        """
        
        # parse cols_format
        len_pos = len([col for col in cols_format if col['tagger'] == 'pos'])
        len_chunk = len([col for col in cols_format if col['tagger'] == 'chunk'])
        len_ner = len([col for col in cols_format if col['tagger'] == 'ner'])

        # set doc_separator
        doc_separator = ' '.join(['-DOCSTART-'] + ['-X-'] * len_pos + ['O'] * len_chunk + ['O'] * len_ner) + '\n'  # TODO: consider the order given by cols_format
        
        
        docs, sentences, tokens = [], [], []
        with open(filepath) as f:

            # parse conll formatted txt file to ConllParser class
            for line in f:
                if line == doc_separator:  # -DOCSTART-|: end of a document
                    if sentences:  # if not empty document
                        docs.append(sentences)
                        sentences = []
                elif not line.strip():  # blank line: the end of a sentence
                    if tokens:  # if not empty sentence (like after -DOCSTART-)
                        sentences.append(tokens)
                        tokens = []
                else:  # inside a sentence: every token
                    a = line.split()
                    poss = {col['type']: ConllPosTag(a[col['col_num']])
                            for col in cols_format if col['tagger'] == 'pos'} if len_pos else None
                    chunks = {col['type']: ConllChunkTag(a[col['col_num']])
                              for col in cols_format if col['tagger'] == 'chunk'} if len_chunk else None
                    ners = {col['type']: ConllNERTag(a[col['col_num']])
                            for col in cols_format if col['tagger'] == 'ner'} if len_ner else None
                    tokens.append({'text': a[0], 'poss': poss, 'chunks': chunks, 'ners': ners})
            if sentences:  # for the last document (without -DOCSTART- at the end)
                docs.append(sentences)
        return docs
    
    @staticmethod
    def add_back_refs_for_md_docs(docs: List[Document]) -> None:  # TODO: Maybe it can be moved to sentence or other places?
        for doc in docs:
            for sent in doc:
                sent.set_document(doc)
                for tok in sent:
                    tok.set_sentence(sent)
                try:
                    for em in sent.entity_mentions_dict.values():
                        em.set_sentence(sent)
                except AttributeError:
                    pass  # to remove
                        
    
    @staticmethod
    def build_md_docs_from_tok_dicts(docs: List[List[List[dict]]]) -> List[Document]:  
        # TODO: Maybe it can further be abstracted such that it can be put in other classes than ConllParser?
        """build the multi-document class from token dictionarys, which are like
        input:
        docs -  [
                    [  # first doc
                        [ tok_dict1, tok_dict2, ...], # first sent
                        [...],
                        ...
                    ], 
                    [...],
                    ...
                ]
                where
                tok_dict = {
                    'text': 'Obama',
                    'poss': {'predict': 'NN', 'gold': 'NN'},
                    'chunks': {'predict': 'I-NNP', 'gold': 'I-NNP'},
                    'ners': {'predict': 'I-PER', 'gold': 'I-PER'}
                    }
        """
            
        did, sid, tid = -1, -1, -1

        docs_classes = []

        for doc in docs:
            did += 1
            sid = -1
            sents_classes = []
            for sent in doc:
                sid += 1
                tid = -1
                toks_classes = []
                for tok in sent:
                    tid += 1
                    toks_classes.append(ConllParser.build_conll_token_from_tok_dicts(tok, tid, sid, did))
                sents_classes.append(Sentence(toks_classes))
            docs_classes.append(Document(sents_classes))
        
        return docs_classes
    
    
    
    @staticmethod
    def build_conll_token_from_tok_dicts(tok, tid, sid, did) -> ConllToken:
        """build a ConllToken from token dictionarys, which are like
        tok_dict = {
        'text': 'Obama',
        'poss': {'predict': 'NN', 'gold': 'NN'},
        'chunks': {'predict': 'I-NNP', 'gold': 'I-NNP'},
        'ners': {'predict': 'I-PER', 'gold': 'I-PER'}
        }
        """
        return ConllToken(tok['text'], poss=tok['poss'], chunks=tok['chunks'], ners=tok['ners'], id_=tid, sid=sid, did=did)
            
    def set_entity_mentions(self) -> None:
        """chunk entity mentions for all sources (i.e. predict, gold)
        effect: call sentence.set_entity_mentions function for all sentences in the parser.docs
        """
        for doc in self.docs:
            for sentence in doc.sentences:
                sentence.set_entity_mentions([src['type'] for src in self.cols_format])

        # set different types of entity mentions from different sources
        _types = ['PER', 'LOC', 'ORG', 'MISC']
        for source in [src['type'] for src in self.cols_format]:
            preffix = 'pred' if source == 'predict' else 'gold'
            ems_attr_name = preffix[0] + 'ems'  # 'pems'/'gems'
            attr_names = [preffix + '_' + _type.lower() + 's' for _type in _types]  # self.pred_pers/self.gold_miscs/...
            
            for attr_name in attr_names:
                self.__setattr__(attr_name, [])
            for doc in self.docs:
                for sentence in doc.sentences:
                    try:
                        for entity_mention in sentence.entity_mentions_dict[source]:
                            attr_name = preffix + '_' + entity_mention.type.lower() + 's'
                            self.__getattribute__(attr_name).append(entity_mention)  # set self.pred_pers/self.gold_miscs/...
                    except KeyError:
                        pass
            
            self.__setattr__(ems_attr_name, [])
            for attr_name in attr_names:
                self.__setattr__(ems_attr_name, self.__getattribute__(ems_attr_name) + self.__getattribute__(attr_name))  # set self.gems/self.pems
        
    @staticmethod
    def set_errors(parser, gold_src, predict_src):  # FIXME: set_errors_xx() duplicated method with methods in NERErrorAnnotator
        for doc in parser.docs:
            ConllParser.set_errors_in_document(doc, gold_src, predict_src)

    @staticmethod
    def set_errors_in_document(doc, gold_src, predict_src):
        for sentence in doc.sentences:
            ConllParser.set_errors_in_sentence(sentence, gold_src, predict_src)

    @staticmethod
    def set_errors_in_sentence(sentence: Sentence, gold_src, predict_src) -> None:
        """
        effect: set sentence.ems_pairs, sentence.ner_results, sentence.corrects, sentence.errors
        """
        sentence.ems_pairs: Union['EntityMentionsPairs', None] = ConllParser.get_pairs(sentence, gold_src, predict_src)  # FIXME: No get_pairs method any more. 
        sentence.ner_results: List[Union['NERErrorComposite', 'NERCorrect']] = None if sentence.ems_pairs is None else \
            sentence.ems_pairs.results

        sentence.set_corrects_from_pairs(sentence.ems_pairs)
        sentence.set_errors_from_pairs(sentence.ems_pairs)
        # TODO: unify the setter or property usage

    def obtain_statistics(self, entity_stat=False, source=None, debug=False):
        _types = ['PER', 'LOC', 'ORG', 'MISC']
        
        print(f'---{self.filepath} ({source})---')
        print('Document number: ', len([doc for doc in self.docs]))
        print('Sentence number: ', len([sen for doc in self.docs for sen in doc.sentences]))
        print('Token number: ', len([token for doc in self.docs for sen in doc.sentences for token in sen.tokens]))

        if source is None:
            source = self.TAGGERSOURCE

        if entity_stat:
            self.set_entity_mentions()
            ent_tot = len(self.__getattribute__(source[0] + 'ems'))
            for type_ in _types:
                ems = self.__getattribute__(source[:4] + '_' + type_.lower() + 's')
                print(type_ + ': {} ({:.0%})'.format(len(ems), len(ems)/ent_tot))
            print('TOTAL: {}'.format(ent_tot))

        if debug:
            print('Empty document number: ', len([doc for doc in self.docs if not doc]))
            print('Empty sentence number: ', len([sen for doc in self.docs for sen in doc.sentences if not sen]))
            print('Empty token number: ', len([token for doc in self.docs
                                               for sen in doc.sentences
                                               for token in sen.tokens if not token]))
            print('Empty-content token number: ', len([token for doc in self.docs for sen in doc.sentences
                                                       for token in sen.tokens if not len(token.text)]))
            print('Empty-id token number: ', len([token for doc in self.docs for sen in doc.sentences
                                                  for token in sen.tokens if token.id is None]))
            print('Token id error number: ',  [(sen.tokens[-1].id, len(sen.tokens)-1) for doc in self.docs
                                               for sen in doc.sentences if sen.tokens[-1].id != len(sen.tokens)-1])
            print('Fullid of empty sentences: ', [sen.fullid for doc in self.docs
                                                  for sen in doc.sentences if not sen])

#             print('Fullid of tokens: ', [token.fullid for doc in self.docs
#             for sen in doc.sentences
#             for token in sen.tokens])

    def filter_results(self, type_):  # TODO: move to NERErrorAnalyzer
        arr = []
        total = 0
        for doc in self.docs:
            for sentence in doc:
                total += 1
                if sentence.ems_pairs:
                    for error in sentence.ems_pairs.errors:
                        if error and error.type['false_error'] == type_:
                            arr.append(error)
        return arr

    def add_filtered_to_dict(self, type_: str, stat: Dict) -> None:  # TODO: move to NERErrorAnalyzer
        stat.update({type_: self.filter_results(type_)})

    def error_overall_stats(self) -> None:  # TODO: move to NERErrorAnalyzer

        # count all errors
        correct_total = 0
        error_total = 0
        ol_total = 0
        sen_total = 0
        for doc in self.docs:
            for sentence in doc:
                sen_total += 1
                if sentence.ems_pairs:
                    ol_total += 1
                    for corr in sentence.ner_corrects:
                        if corr:
                            correct_total += 1
                    for error in sentence.ner_errors:
                        if error:
                            error_total += 1
        print('---Overall Results---')
        print('found entity mentions:', len(self.pems))
        print('true entity mentions:', len(self.gems))
        print('correct_total: ', correct_total)
        print('error_total: ', error_total)
        self.precision = correct_total/len(self.pems)
        self.recall = correct_total/len(self.gems)
        self.macrof1 = 2*self.precision*self.recall/(self.recall+self.precision)
        print('precision: {:.2%}'.format(self.precision))
        print('recall: {:.2%}'.format(self.recall))
        print('macro-f1: {:.2%}'.format(self.macrof1))
        print('corrects ratio: {:.2%}'.format(correct_total/(correct_total+error_total)))
        print('all corrects and errors', correct_total + error_total)
        print('the number of sentences with/without entities (predict + gold): {} ({:.0%}), {} ({:.0%})'.format(
            ol_total, ol_total/sen_total, sen_total - ol_total, (sen_total - ol_total)/sen_total))

    def error_type_stats(self):  # TODO: move to NERErrorAnalyzer

        # stat = {}
        # for type_ in NERErrorComposite.all_span_error_types:
        #     self.add_filtered_to_dict(type_, stat)
        #
        pass

    def print_all_errors(self) -> None:  # TODO: move to NERErrorAnalyzer
        for doc in self.docs:
            for sentence in doc:
                if sentence.ner_errors:
                    for error in sentence.ner_errors:
                        print(str(error))

    def print_corrects(self, num=10):  # TODO: move to NERErrorAnalyzer
        count = 0
        for doc in self.docs:
            for sentence in doc:
                count += 1
                if count % num == 0:
                    sentence.print_corrects()

    def confusion_matrix(self):  # TODO: move to NERErrorAnalyzer
        pass

    def get_from_fullid(self, fullid):  # TODO: move to ConllDocuments
        abbrs = {
            'doc': 'D',
            'sen': 'S',
            'token': 'T',
            'em': 'EM',
            'em_ol': 'OL',
            'ner_error': 'NERErr',
            'ner_correct': 'NERCorr'
        }

    def get_doc_from_did(self, did) -> Document:  # TODO: remove this
        return self.docs[did]

    def get_sen_from_sid(self, did, sid) -> Sentence:  # TODO: remove this
        return self.get_doc_from_did(did).sentences[sid]
    
    
class EntityMentionAnnotator:
    # put set_entity_mentions() here
    pass

if __name__ == '__main__':

    import os.path
    import doctest

    failure_count, test_count = doctest.testmod()
#     doctest.run_docstring_examples(ConllParser.get_entity_mentions_from_sent, globals())
    if failure_count == 0:
        
        print('{} tests passed!!!!!!'.format(test_count))
    
        path = os.path.dirname(os.path.abspath(__file__)) + '/../test/train.pred.gold'
        train_parser = ConllParser(path)
        train_parser.obtain_statistics(entity_stat=True, source='predict')
        train_parser.set_entity_mentions()
