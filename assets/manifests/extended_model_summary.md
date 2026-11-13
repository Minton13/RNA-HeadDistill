# Extended Model Summary

| model | size GiB | download | old forward | old attention | new forward | new attention | layers | heads | note |
|---|---:|---|---|---|---|---|---:|---:|---|
| rnaernie | 0.6458 | skipped_existing | True | True | True | True | 12.0 | 12.0 | RNA-ERNIE MultiMolecule model; attention teacher candidate |
| ernierna | 0.6429 | skipped_existing | None | None | None | None | None | None | ERNIE-RNA MultiMolecule model; attention teacher candidate |
| rnamsm | 0.7149 | ok | True | False | False | False | 10 | 12 | RNA-MSM MultiMolecule model; attention teacher candidate |
| caduceus_ps_1k_d118 | 0.0018 | ok | None | None | False | False | None | None | Caduceus DNA bidirectional long-context model; hidden/control candidate |
| caduceus_ph_1k_d118 | 0.0018 | ok | None | None | False | False | None | None | Caduceus DNA reverse-complement equivariant candidate |
| caduceus_ps_131k_d256 | 0.0289 | ok | None | None | False | False | None | None | Small-footprint Caduceus long-context checkpoint |
| gena_lm_bert_base | 0.5077 | ok | True | True | True | True | 12.0 | 12.0 | GENA-LM BERT-base DNA model; likely attention-capable |
| gena_lm_bigbird_base_t2t | 0.4261 | ok | True | False | False | False | 12 | 12 | GENA-LM BigBird DNA model; long-context hidden/control candidate |
| carbon_500m | 0.9695 | ok | True | True | True | True | 28.0 | 16.0 | Newer genomic foundation model; server/local capability test required |
| evo2_1b_base | 2.513 | ok | None | None | False | False | None | None | Evo2 1B base weights; not expected to load via vanilla AutoModel without Evo2 runtime |
| ernierna_ss | 0.6475 | ok | None | None | None | None | None | None | ERNIE-RNA secondary-structure variant |
