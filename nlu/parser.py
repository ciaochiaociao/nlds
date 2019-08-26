from nlu.data import *


class ConllParser:
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
        >>> path = os.path.dirname(__file__) + traintrain.pred.gold>>> train_parser = ConllParser(path)
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

        # parse cols_format
        len_pos = len([col for col in self.cols_format if col['tagger'] == 'pos'])
        len_chunk = len([col for col in self.cols_format if col['tagger'] == 'chunk'])
        len_ner = len([col for col in self.cols_format if col['tagger'] == 'ner'])

        # set doc_separator
        doc_separator = ' '.join(['-DOCSTART-'] + ['-X-'] * len_pos + ['O'] * len_chunk + ['O'] * len_ner) + '\n'

        # local variables
        sentences = []
        tokens = []

        count = 0
        did = 0
        sid = 0
        tid = 0
        with open(self.filepath) as f:

            # parse conll formatted txt file to ConllParser class
            for line in f:
                count += 1
                if line == doc_separator:  # -DOCSTART-|: end of a document
                    if sentences:  # if not empty document
                        doc = Document(sentences)
                        self.docs.append(doc)
                        for sentence in sentences:
                            sentence.set_document(doc)
                        sentences = []
                        did += 1
                    sid = 0
                elif not line.strip():  # blank line: the end of a sentence
                    if tokens:  # if not empty sentence (like after -DOCSTART-)
                        sent = Sentence(tokens)
                        sentences.append(sent)
                        for token in tokens:
                            token.set_sentence(sent)
                        tokens = []
                        sid += 1
                    tid = 0
                else:  # inside a sentence: every token
                    a = line.split()
                    poss = {col['type']: ConllPosTag(a[col['col_num']])
                            for col in self.cols_format if col['tagger'] == 'pos'} if len_pos else None
                    chunks = {col['type']: ConllChunkTag(a[col['col_num']])
                              for col in self.cols_format if col['tagger'] == 'chunk'} if len_chunk else None
                    ners = {col['type']: ConllNERTag(a[col['col_num']])
                            for col in self.cols_format if col['tagger'] == 'ner'} if len_ner else None
                    tokens.append(ConllToken(a[0], poss=poss, chunks=chunks, ners=ners, id_=tid, sid=sid, did=did))
                    tid += 1
            if sentences:  # for the last document (without -DOCSTART- at the end)
                self.docs.append(Document(sentences))

    def set_entity_mentions(self) -> None:
        
        for doc in self.docs:
            for sentence in doc.sentences:
                self.set_entity_mentions_for_one_sentence(sentence, [src['type'] for src in self.cols_format])

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
    def set_entity_mentions_for_one_sentence(sentence: Sentence, sources: List) -> None:

        for source in sources:

            tokens_tray: List[Token] = []
            entity_mentions: List[EntityMention] = []
    #         last_token = ConllToken(text='', ner='O')
            last_token = None
            eid = 0

            for i, token in enumerate(sentence.tokens):
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

            if entity_mentions:
                sentence.entity_mentions_dict[source] = entity_mentions

    @staticmethod
    def set_errors(parser, gold_src, predict_src):
        for doc in parser.docs:
            ConllParser.set_errors_in_document(doc, gold_src, predict_src)

    @staticmethod
    def set_errors_in_document(doc, gold_src, predict_src):
        for sentence in doc.sentences:
            ConllParser.set_errors_in_sentence(sentence, gold_src, predict_src)

    @staticmethod
    def set_errors_in_sentence(sentence: Sentence, gold_src, predict_src) -> None:

        sentence.ems_pairs: Union['EntityMentionsPairs', None] = ConllParser.get_pairs(sentence, gold_src, predict_src)
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

    def filter_results(self, type_):
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

    def add_filtered_to_dict(self, type_: str, stat: Dict) -> None:
        stat.update({type_: self.filter_results(type_)})

    def error_overall_stats(self) -> None:

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

    def error_type_stats(self):

        # stat = {}
        # for type_ in NERErrorComposite.all_span_error_types:
        #     self.add_filtered_to_dict(type_, stat)
        #
        pass

    def print_all_errors(self) -> None:
        for doc in self.docs:
            for sentence in doc:
                if sentence.ner_errors:
                    for error in sentence.ner_errors:
                        print(str(error))

    def print_corrects(self, num=10):
        count = 0
        for doc in self.docs:
            for sentence in doc:
                count += 1
                if count % num == 0:
                    sentence.print_corrects()

    def confusion_matrix(self):
        pass

    def get_from_fullid(self, fullid):  # TODO
        abbrs = {
            'doc': 'D',
            'sen': 'S',
            'token': 'T',
            'em': 'EM',
            'em_ol': 'OL',
            'ner_error': 'NERErr',
            'ner_correct': 'NERCorr'
        }

    def get_doc_from_did(self, did) -> Document:
        return self.docs[did]

    def get_sen_from_sid(self, did, sid) -> Sentence:
        return self.get_doc_from_did(did).sentences[sid]


if __name__ == '__main__':

    import os.path
    import doctest

    doctest.testmod()
    path = os.path.dirname(__file__) + '/../test/train.pred.gold'
    train_parser = ConllParser(path)
    train_parser.obtain_statistics(entity_stat=True, source='predict')
    train_parser.set_entity_mentions()
