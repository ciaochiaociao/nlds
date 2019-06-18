# NER

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

## Common Attributes

- pems
- gems
- ptkns
- gtkns

