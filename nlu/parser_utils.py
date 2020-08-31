import copy


def bioes2iob1(ner, last):
    
    def startwith(ner, prefix):
        return ner[0] == prefix

    def change_pre(ner, prefix):
        return prefix + ner[1:]

    def isend(ner):
        return startwith(ner, 'E') or startwith(ner, 'S')

    def isstart(ner):
        return startwith(ner, 'B') or startwith(ner, 'S')

    def isoutside(ner):
        return startwith(ner, 'O')

    def issametype(ner1, ner2):
        return ner1[2:] == ner2[2:]
    
    if isend(last) and isstart(ner) and issametype(last, ner):  # Boundary
        newner = change_pre(ner, 'B')
    elif not isoutside(ner):
        newner = change_pre(ner, 'I')
    else:
        newner = ner
    return newner


def bioes2iob1_list(ners):
    last = 'O'
    newners = []
    for ner in ners:
        newner = bioes2iob1(ner, last)
        last = ner
        newners.append(newner)
    return newners


def bioes2iob1_file(infile, outfile, bieos_cols=[1,2]):
    """
    Washington I-PER I-NNP
    ...
    
    bieos_cols: ex- [1,2]
    >>> bioes2iob1_file('test/wnut.test.gold.pred', 'test/wnut.test.gold.pred.iob1test')
    """
    if isinstance(infile, str):
        infile = open(infile, encoding='utf-8')
    if isinstance(outfile, str):
        outfile = open(outfile, 'w', encoding='utf-8')

    # add encoding='utf-8' to support windows OS
    lastcols = ['O'] * (max(bieos_cols) + 1)
    for line in infile:
        if line.strip():

            cols = line.split()
            newcols = copy.copy(cols)
            for bieos_col in bieos_cols:
                newcols[bieos_col] = bioes2iob1(cols[bieos_col], lastcols[bieos_col])

            lastcols = cols
            outfile.write(' '.join(newcols) + '\n')
        else:
            outfile.write(line)
            lastcols = ['O'] * (max(bieos_cols) + 1)
    infile.close()
    outfile.close()


def bioes2iob2(tag):
    return tag.replace('S-', 'B-').replace('E-', 'I-')


def bioes2iob2_file(infile, outfile, bieos_cols=[1, 2]):  # deprecated: combined into convert_scheme

    if isinstance(infile, str):
        infile = open(infile, encoding='utf-8')
    if isinstance(outfile, str):
        outfile = open(outfile, 'w', encoding='utf-8')

    # add encoding='utf-8' to support windows OS
    for line in infile:
        if line.strip():
            cols = line.split()
            newcols = copy.copy(cols)
            for bieos_col in bieos_cols:
                newcols[bieos_col] = bioes2iob2(cols[bieos_col])

            outfile.write(' '.join(newcols) + '\n')
        else:
            outfile.write(line)
    infile.close()
    outfile.close()


def iob12bioes_file(*args, **kwargs):
    return iob2bioes_file(*args, **kwargs, convert_fn='iob12bioes')


def iob22bioes_file(*args, **kwargs):
    return iob2bioes_file(*args, **kwargs, convert_fn='iob22bioes')


def iob2bioes_file(infile, outfile, col_nums=[1,2], col_sep=' ', doc_sep_tok='-DOCSTART-', convert_fn=None):

    if convert_fn is None or convert_fn == 'iob12bioes':
        convert_fn = iob12bioes
    elif convert_fn == 'iob22bioes':
        convert_fn = iob22bioes
    else:
        raise ValueError

    cols_format = [{'type': 'tmp'+str(col_num), 'col_num': col_num, 'tagger': 'ner'} for col_num in col_nums]
    # e.g. [{'type': 'predict', 'col_num': 1, 'tagger': 'ner'},
    # {'type': 'gold', 'col_num': 2, 'tagger': 'ner'}]

    docs = parse_conll_to_tok_dicts(infile, cols_format, doc_sep_tok=doc_sep_tok, col_sep=col_sep)

    if isinstance(outfile, str):
        outfile = open(outfile, 'w', encoding='utf-8')

    for doc in docs:
        outfile.write(doc_sep_tok + col_sep.join(['O'] * len(col_nums)) + '\n\n')
        for sent in doc:
            # convert tag schemes for each tag columns
            new_tags_list = []
            for col_num in col_nums:
                tags = [token['ners']['tmp'+str(col_num)] for token in sent]
                new_tags = convert_fn(tags)
                new_tags_list.append(new_tags)
            assert all([len(_l) == len(sent) for _l in new_tags_list])

            # write out
            for token, *token_new_tags in zip(sent, *new_tags_list):
                outfile.write(col_sep.join([token['text']] + token_new_tags) + '\n')
            outfile.write('\n')
    outfile.close()


def iob12bioes(tags):
    return iob22bioes(iob12iob2(tags))


def iob12iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    .. seealso:: `https://github.com/flairNLP/flair/blob/master/flair/data.py <https://github.com/flairNLP/flair/blob/master/flair/data.py>`_.
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == "O":
            new_tags.append(tag)
            continue

        # validity check (the case which is so invalid that it can not be fixed)
        split = tag.split("-")
        if len(split) != 2 or split[0] not in ["I", "B"]:
            # (cwhsu) check 'X-X' and 'B-, I-' # FIXME: len(split) == 3 when using wnut, e.g., I-creative-work
            return False

        if split[0] == "B":
            new_tags.append(tag)
            continue
        elif i == 0 or tags[i - 1] == "O":  # conversion IOB1 to IOB2
            # (cwhsu) case: 'I-X'; when the left boundary is detected (BOE, beginning of an entity)
            new_tags.append("B" + tag[1:])
        elif tags[i - 1][1:] == tag[1:]:
            # (cwhsu) case: 'I-X' and is not BOE; check if the suffix of the previous tag is the same as the current tag
            new_tags.append(tag)
            continue
        else:  # conversion IOB1 to IOB2
            # (cwhsu) case: 'I-X' + not BOE BUT has type mismatch with the previous tag of the same entity
            # => create a new entity
            # e.g. B-ORG I-PER => B-ORG B-PER
            new_tags.append("B" + tag[1:])

    return new_tags


def iob22bioes(tags):
    """
    IOB -> IOBES
    .. seealso:: modified from `https://github.com/flairNLP/flair/blob/master/flair/data.py <https://github.com/flairNLP/flair/blob/master/flair/data.py>`_.
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == "O":
            new_tags.append(tag)
        elif tag.split("-")[0] == "B":
            if i + 1 != len(tags) and tags[i + 1].split("-")[0] == "I":
                # (cwhsu) not reaching the right boundary (not EOE, not End Of an Entity)
                # note -        EOE:    EOS     OR      the next tag is not 'I' ('B-X' or 'O')
                #           not EOE: not EOS    AND     the next tag is     'I' ('B-X' or 'O')
                new_tags.append(tag)
            else:
                # (cwhsu) convert 'B' to 'S' when it's EOE
                new_tags.append(tag.replace("B-", "S-"))
        elif tag.split("-")[0] == "I":
            if i + 1 < len(tags) and tags[i + 1].split("-")[0] == "I":
                new_tags.append(tag)
            else:  # (cwhsu) convert 'I' to 'E' when it's EOE
                new_tags.append(tag.replace("I-", "E-"))
        else:
            raise Exception("Invalid IOB format!")
    return new_tags


# modified from https://github.com/chakki-works/seqeval
def get_entities(seq, suffix=False):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        # from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i-1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    # (cwhsu) prev_tag != 'O' (== 'B', 'I') and tag == 'B', 'S', 'O'
    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        # (cwhsu) e.g. prev_tag == 'B', 'I' and tag == 'I', 'E' (non-singelton) and inconsistent type
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        # (cwhsu) e.g. prev_tag == 'B', 'I' and tag == 'I', 'E' (non-singelton) and inconsistent type
        chunk_start = True

    return chunk_start


if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-s', '--source', type=str)
    argparser.add_argument('--source-file', type=argparse.FileType('r'), default='-')
    argparser.add_argument('-t', '--target', type=str)
    argparser.add_argument('--target-file', type=argparse.FileType('w'), default='-')
    argparser.add_argument('--cols', nargs='*', default=[1, 2], type=int)
    args = argparser.parse_args()
    if args.source in ['bioes', 'iobes'] and args.target == 'iob1':
        bioes2iob1_file(args.source_file, args.target_file, bieos_cols=args.cols)
    elif args.source in ['bioes', 'iobes'] and args.target == 'iob2':
        bioes2iob2_file(args.source_file, args.target_file, args.cols)
    elif args.source == 'iob1' and args.target in ['bioes', 'iobes']:
        iob12bioes_file(args.source_file, args.target_file, col_nums=args.cols)
    elif args.source == 'iob2' and args.target in ['bioes', 'iobes']:
        iob22bioes_file(args.source_file, args.target_file, col_nums=args.cols)
    else:
        raise argparse.ArgumentTypeError('source/target invalid type: ' + args.source + '/' + args.target + ' must be bioes/iob1/iob2')