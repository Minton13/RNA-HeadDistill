# More RNA Model Head Extraction Report

| model | size_gib | tokenizer_ok | model_load_ok | forward_ok | has_attentions | n_layers | n_heads | hidden_size | attn_shape | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mrnafm | 0.8914 | True | True | True | True | 12.0 | 20.0 | 1280.0 | 1x20x7x7 |  |
| plantrna_fm | 0.1271 | False | False | False | False | nan | nan | nan | nan | RuntimeError("multimolecule.AutoTokenizer: StrictDataclassFieldValidationError: Validation error for field 'hidden_dropout_prob':\n    TypeError: Field 'hidden_dropout_prob' with value 0 doesn't match any type in (<class |
| mp_rna | 0.6926 | True | False | False | False | nan | nan | nan | nan | RuntimeError('multimolecule.AutoModel: ImportError: This modeling file requires the following packages that were not found in your environment: ViennaRNA. Run `pip install ViennaRNA`; AutoModel+eager: ImportError: This m |
| aido_rna_25m_mars | 0.0941 | True | False | False | False | nan | nan | nan | nan | RuntimeError("multimolecule.AutoModel: KeyError: 'swiglu'; AutoModel+eager: KeyError: 'swiglu'; AutoModel: KeyError: 'swiglu'; AutoModelForMaskedLM+eager: KeyError: 'swiglu'; AutoModelForMaskedLM: KeyError: 'swiglu'") |
| aido_rna_300m_mars | 1.1187 | True | False | False | False | nan | nan | nan | nan | RuntimeError("multimolecule.AutoModel: KeyError: 'swiglu'; AutoModel+eager: KeyError: 'swiglu'; AutoModel: KeyError: 'swiglu'; AutoModelForMaskedLM+eager: KeyError: 'swiglu'; AutoModelForMaskedLM: KeyError: 'swiglu'") |
| aido_rna_650m | 2.4155 | True | False | False | False | nan | nan | nan | nan | RuntimeError("multimolecule.AutoModel: KeyError: 'swiglu'; AutoModel+eager: KeyError: 'swiglu'; AutoModel: KeyError: 'swiglu'; AutoModelForMaskedLM+eager: KeyError: 'swiglu'; AutoModelForMaskedLM: KeyError: 'swiglu'") |
| aido_rna_1m_mars_taykhoom | 0.0043 | True | False | False | False | nan | nan | nan | nan | RuntimeError('multimolecule.AutoModel: ImportError: This modeling file requires the following packages that were not found in your environment: flash_attn. Run `pip install flash_attn`; AutoModel+eager: ImportError: This |
