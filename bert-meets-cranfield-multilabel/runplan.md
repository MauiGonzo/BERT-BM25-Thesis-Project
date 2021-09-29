plan for runs to for multilabel

__arg_max__
1. with default BCE with Logit Loss
1.1.1 EPOCH=1; LR=2E-05
1.1.2 EPOCH=2; LR=2E-05
1.1.3 EPOCH=1; LR=3E-05
1.1.4 EPOCH=2; LR=3E-05
2. with custom BCEwLL
that is; with the class weights set. 
1.1.1 EPOCH=1; LR=2E-05
1.1.2 EPOCH=2; LR=2E-05
1.1.3 EPOCH=1; LR=3E-05
1.1.4 EPOCH=2; LR=3E-05
3. with MultiLabelMarginLoss
.
4. Multi Task 50/50 ratio
.
5. Multi Taks 80/20 ratio

__per label ranking__
that is taking all the 5 predictions and deriving NDCG performance for the ranking, if you would take label *l*
for now considered as reference experiment, because so far arg_max appears to be better. only with default BCEwithLogitsLoss and the four LR and EPOCH combinations

__regresion__
tests with the various conversion tables

__miscelaneous__
_lowest priority_