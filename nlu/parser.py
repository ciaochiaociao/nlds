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
        >>> path = os.path.dirname(__file__) + '/../rcv1.train.compare2'
        >>> train_parser = ConllParser(path)
        >>> train_parser.obtain_statistics(entity_stat=True, source='predict')  # doctest: +ELLIPSIS
        ---...
        Document number:  946
        Sentence number:  14041
        Token number:  203621
        ... PER: 6600
        ... LOC: 7140
        ... ORG: 6321
        ... MISC: 3438
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

    def obtain_statistics(self, entity_stat=False, source=None, debug=False):
        print(f'---{self.filepath} ({source})---')
        print('Document number: ', len([doc for doc in self.docs]))
        print('Sentence number: ', len([sen for doc in self.docs for sen in doc.sentences]))
        print('Token number: ', len([token for doc in self.docs for sen in doc.sentences for token in sen.tokens]))

        if source is None:
            source = self.TAGGERSOURCE

        if entity_stat:
            per, loc, org, misc = [], [], [], []
            self.set_entity_mentions()

            for doc in self.docs:
                for sentence in doc.sentences:
                    try:
                        for entity_mention in sentence.entity_mentions_dict[source]:
                            if entity_mention.type == 'PER':
                                per.append(entity_mention)
                            elif entity_mention.type == 'LOC':
                                loc.append(entity_mention)
                            elif entity_mention.type == 'ORG':
                                org.append(entity_mention)
                            elif entity_mention.type == 'MISC':
                                misc.append(entity_mention)
                    except KeyError:
                        pass

            print('PER: %s' % (len(per)))
            print('LOC: %s' % (len(loc)))
            print('ORG: %s' % (len(org)))
            print('MISC: %s' % (len(misc)))
            print('TOTAL: %s' % (len(per)+len(loc)+len(org)+len(misc)))

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


if __name__ == '__main__':

    import os.path

    path = os.path.dirname(__file__) + '/../rcv1.train.compare2'
    print(path)
    train_parser = ConllParser(path)
    train_parser.obtain_statistics(entity_stat=True, source='predict')
    train_parser.set_entity_mentions()
