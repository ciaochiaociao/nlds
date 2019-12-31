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


def bioes2iob1_file(infile, outfile, bieos_cols=[1,2]):
    """
    Washington I-PER I-NNP
    ...

    bieos_cols: ex- [1,2]

    >>> bioes2iob1_file('test/wnut.test.gold.pred', 'test/wnut.test.gold.pred.iob1test')
    """
    with open(infile, encoding='utf-8') as f, open(outfile, 'w', encoding='utf-8') as fo:
    # add encoding='utf-8' to support windows OS
        lastcols = ['O'] * (max(bieos_cols) + 1)
        for line in f:
            if line.strip():
                
                cols = line.split()
                newcols = copy.copy(cols)
                for bieos_col in bieos_cols:
                    newcols[bieos_col] = bioes2iob1(cols[bieos_col], lastcols[bieos_col])
    
                lastcols = cols
                fo.write(' '.join(newcols) + '\n')
            else:
                fo.write(line)
