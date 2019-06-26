# NER Error Analyzer 

## Quick Start
```python
from nlu.error import *
from nlu.parser import *


cols_format = [{'type': 'predict', 'col_num': 1, 'tagger': 'ner'},
                {'type': 'gold', 'col_num': 2, 'tagger': 'ner'}]

parser = ConllParser('rcv1.testb.compare2', cols_format)

parser.obtain_statistics(entity_stat=True, source='predict')

parser.obtain_statistics(entity_stat=True, source='gold')

NERErrorAnnotator.annotate(parser)

parser.print_corrects()

parser.print_all_errors()

parser.error_overall_stats()
```

see the section Input Format below to know what the input format is

## Usage
### import
```python
from nlu.error import *
from nlu.parser import *
```
### Create a `ConllParser` instance first with the input of the file path with specifying the column number in `cols_format` field
`ConllParser(filepath)`

```python
cols_format = [{'type': 'predict', 'col_num': 1, 'tagger': 'ner'},
                {'type': 'gold', 'col_num': 2, 'tagger': 'ner'}]

parser = ConllParser('rcv1.testb.compare2', cols_format)
```
### obtain the basic statistics by `obtain_statistics()` method
```python
parser.obtain_statistics(entity_stat=True, source='predict')

parser.obtain_statistics(entity_stat=True, source='gold')
```
### To "Annotate" NER Errors in the documents inside ConllParser
`NERErrorAnnotator.annotate(parser)`

### To print out all corrects/errors, use
`parser.print_corrects()` or
`parser.print_all_errors()`

or use the function `error_overall_stats()` method to get the stats

## Input File Format

The input file format of `ConllParser` is following the column format used by Conll03.

For example,

```text
Natural I-ORG O
Language I-ORG O
Laboratory I-ORG I-ORG
...
```

where the first column is the text, the second and the third are the predicted and the ground truth tag respectively, where the order can be specified in the keyword `cols_format` in `ConllParser` in instantialization:
 ```
 cols_format = [{'type': 'predict', 'col_num': 1, 'tagger': 'ner'},
                {'type': 'gold', 'col_num': 2, 'tagger': 'ner'}]  # col_num starts from 0
```
I recommend to use shell command `awk '{print $x}' filepath` to obtain the x-th column, like `awk '{print $4} filepath'` to obtain the 4-th column.

And use `paste file1.txt file2.txt` to concatenate two files.

For example,

```bash
awk '{print $4}' eng.train > ner_tags_file  # $num starts from 1
paste ner_pred_tags_file ner_tags_file
```


## Types of Span Errors

Types | Number of Mentions (Predicted and Gold) | Subtypes | Examples| Notes 
:---: | :---: | --- |----- |----- 
Missing Mention<br />(False Negative) | 1 | TYPES&rightarrow;O |[] &rightarrow; None # todo|
Extra Mention<br />(False Positive) | 1 | O&rightarrow;TYPES | None &rightarrow; [...]  # todo |  
Mention with Wrong Type<br />(Type Errors) | &ge; 2 | TYPES-> TYPES - self<br />( {(p, g) \| p &isin; T, g &isin; T - p } ) | [<sub>PER</sub>...] &rightarrow; [<sub>ORG</sub>...] # todo | But the spans are the same 
Missing Tokens | 2 | L/ R/ LR Diminished | \[<sub>MISC</sub>1991 World Cup] &rightarrow; \[<sub>MISC</sub>1991] \[<sub>MISC</sub> World Cup] | also possible with type errors 
Extra Tokens | 2 | L/R/LR Expanded | [...] &rightarrow; [......]  # todo| also possible with type errors 
Missing + Extra Tokens | 2 | L/R Crossed | ..[...].. &rightarrow; .[..]... | also possible with type errors 
Conflated Mention | &ge; 3 |  |  \[]\[]\[] &rightarrow; \[]  # todo | also possible with type errors 
Divided Mention | &ge; 3 |  | [<sub>MISC</sub>1991 World Cup] &rightarrow; \[<sub>MISC</sub>1991] \[<sub>MISC</sub> World Cup]<br />[<sub>PER</sub>Barack Hussein Obama] &rightarrow; \[<sub>PER</sub>Barack]\[<sub>PER</sub>Hussein]\[<sub>PER</sub>Obama] | also possible with type errors 
Complicated Case | &ge; 3 |                                                              | \[]\[]\[] &rightarrow; \[]\[]  # todo | also possible with type errors 
Ex - <br />Mention with Wrong Segmentation<br />(Same overall range but wrong segmentation) | &ge; 4 |                                                              | \[...]\[......]\[.] &rightarrow; \[......][.....] | also possible with type errors 
