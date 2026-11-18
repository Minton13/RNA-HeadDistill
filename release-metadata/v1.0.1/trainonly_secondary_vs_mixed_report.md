# Train-only secondary re-ranking vs stored mixed-split selection

- train-only atlas rows (cleaned): 6604 (28 models)
- re-derived top-30 vs stored top-30 overlap: 25/30
- heads only in train-only top-30: 5; only in stored: 5

## Fate of the two L12-route heads selected from this task
- rinalmo_giga L31H13: stored rank 1 (auprc 0.9111) | train-only rank 1 (auprc 0.9810)
- rinalmo_giga L31H1: stored rank 2 (auprc 0.8806) | train-only rank 2 (auprc 0.9728)

## Top-10 of the re-derived train-only ranking

        model  layer  head  metric_primary
 rinalmo_giga     31    13        0.980978
 rinalmo_giga     31     1        0.972791
 rinalmo_mega     29    12        0.971728
 rinalmo_mega     28    10        0.971207
 rinalmo_mega     29     3        0.964314
 rinalmo_giga     32    12        0.962635
rinalmo_micro     11     8        0.914760
        rnafm      5    11        0.901831
rinalmo_micro     11     3        0.896381
     rnaernie     11     0        0.885719

## Heads entering/leaving the top-30

entering: [('moderngena_base', 12, 11), ('moderngena_base', 15, 6), ('rinalmo_giga', 32, 16), ('rinalmo_micro', 10, 13), ('rnaernie', 5, 11)]

leaving: [('rinalmo_giga', 31, 7), ('rinalmo_micro', 11, 0), ('rnabert', 0, 5), ('rnabert', 1, 0), ('rnabert', 3, 7)]