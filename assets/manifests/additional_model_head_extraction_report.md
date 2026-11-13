# Additional Model Head Extraction Report

| model | type | size_gib | tokenizer_ok | model_load_ok | forward_ok | has_attentions | n_layers | n_heads | hidden_size | attn_shape | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| utrbert_4mer | rna | 0.3247 | True | True | True | True | 12.0 | 12.0 | 768.0 | 1x12x14x14 |  |
| utrbert_5mer | rna | 0.3318 | True | True | True | True | 12.0 | 12.0 | 768.0 | 1x12x13x13 |  |
| utrbert_6mer | rna | 0.3677 | True | True | True | True | 12.0 | 12.0 | 768.0 | 1x12x12x12 |  |
| utrlm_te_el | rna | 0.0046 | True | True | True | True | 6.0 | 16.0 | 128.0 | 1x16x17x17 |  |
| rnafm_ss | rna | 0.3801 | True | False | False | False | nan | nan | nan | None | RuntimeError('multimolecule.AutoModel: ValueError: vocab_size (28) must be 26 when codon=False.; AutoModel+eager: ValueError: vocab_size (28) must be 26 when codon=False.; AutoModel: ValueError: vocab_size (28) must be 2 |
| gena_lm_bert_base_t2t | dna | 0.5054 | True | True | True | True | 12.0 | 12.0 | 768.0 | 1x12x6x6 |  |
| gena_lm_bert_base_lastln_t2t | dna | 0.5054 | True | True | True | True | 12.0 | 12.0 | 768.0 | 1x12x6x6 |  |
| gena_lm_bert_base_t2t_multi | dna | 0.5054 | True | True | True | True | 12.0 | 12.0 | 768.0 | 1x12x6x6 |  |
| gena_lm_bigbird_base_sparse | dna | 0.5107 | True | True | True | True | 12.0 | 12.0 | 768.0 | 1x12x6x6 |  |
| moderngena_base | dna | 0.5085 | True | True | True | False | 0.0 | nan | 768.0 | None |  |
| nt_v2_50m_multi | dna | 0.4128 | True | False | False | False | nan | nan | nan | None | RuntimeError("multimolecule.AutoModel: AttributeError: 'EsmConfig' object has no attribute 'rope_theta'; AutoModel+eager: AttributeError: 'EsmConfig' object has no attribute 'rope_theta'; AutoModel: AttributeError: 'EsmC |
| nt_v2_50m_3mer_multi | dna | 0.382 | True | False | False | False | nan | nan | nan | None | RuntimeError("multimolecule.AutoModel: AttributeError: 'EsmConfig' object has no attribute 'rope_theta'; AutoModel+eager: AttributeError: 'EsmConfig' object has no attribute 'rope_theta'; AutoModel: AttributeError: 'EsmC |
| dnabert_3 | dna | 0.323 | True | True | True | False | 0.0 | nan | 768.0 | None |  |
| dnabert_4 | dna | 0.3236 | True | True | True | False | 0.0 | nan | 768.0 | None |  |
| dnabert_5 | dna | 0.3258 | True | True | True | False | 0.0 | nan | 768.0 | None |  |
| dnabert_6 | dna | 0.3346 | True | True | True | False | 0.0 | nan | 768.0 | None |  |
| dnabert_s | dna | 0.4364 | True | False | False | False | nan | nan | nan | None | RuntimeError('multimolecule.AutoModel: RuntimeError: Tensor on device meta is not on the expected device cpu!; AutoModel+eager: RuntimeError: Tensor on device meta is not on the expected device cpu!; AutoModel: RuntimeEr |
| mrnabert_yyly66 | rna | 0.4248 | True | False | False | False | nan | nan | nan | None | RuntimeError("multimolecule.AutoModel: RuntimeError: Tensor on device meta is not on the expected device cpu!; AutoModel+eager: RuntimeError: Tensor on device meta is not on the expected device cpu!; AutoModel: RuntimeEr |
