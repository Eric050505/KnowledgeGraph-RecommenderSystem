## Project of SUSTech CS303: Artificial Intelligence, Fall 2024

Knowledge Graph (KG)-based recommender system technology uses knowledge graphs 
as auxiliary information to improve the accuracy and explain-ability of the result of 
recommendations. Knowledge graphs, which are graphs with nodes representing 
entities and edges representing relationships, help to illustrate the relationships between items 
and their attributes, and integrate user and user-side information, capturing the relationships 
between users and items, as well as user preferences, more accurately. 

### Quick Start
run  `python eval/evaluate.py`

### Code Explanation
* `data`: 
  * Original data to build an initial knowledge graph.
  * Positive and negative samples for training.

* `eval`: Including training and testing the KGRS.

* `model`: KGRS model source code.

* `test`: Some performance records while tuning.
