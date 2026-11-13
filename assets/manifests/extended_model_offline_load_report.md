# RNAscope Model Offline Load Report

- generated_at: `2026-06-16T22:11:05`

| model | type | priority | size_gib | tokenizer | model | forward | attentions | hidden |
|---|---|---:|---:|---|---|---|---|---|
| `rnaernie` | `rna` | `required_extension` | `0.6458` | `True` | `True` | `None` | `None` | `None` |
| `ernierna` | `rna` | `required_extension` | `0.6429` | `True` | `False` | `False` | `False` | `False` |
| `rnamsm` | `rna` | `required_extension` | `0.7149` | `True` | `True` | `None` | `None` | `None` |
| `caduceus_ps_1k_d118` | `dna` | `required_extension` | `0.0018` | `True` | `False` | `False` | `False` | `False` |
| `caduceus_ph_1k_d118` | `dna` | `required_extension` | `0.0018` | `True` | `False` | `False` | `False` | `False` |
| `caduceus_ps_131k_d256` | `dna` | `required_extension` | `0.0289` | `True` | `False` | `False` | `False` | `False` |
| `gena_lm_bert_base` | `dna` | `required_extension` | `0.5077` | `True` | `True` | `None` | `None` | `None` |
| `gena_lm_bigbird_base_t2t` | `dna` | `required_extension` | `0.4261` | `True` | `True` | `None` | `None` | `None` |
| `carbon_500m` | `dna` | `required_extension` | `0.9695` | `False` | `False` | `False` | `False` | `False` |
| `evo2_1b_base` | `dna` | `required_extension` | `2.513` | `False` | `False` | `False` | `False` | `False` |

## ernierna
```text
AutoModel: ValueError: Vocab size in MultiMolecule RNA alphabet (26) does not match the one in config (28) | AutoModel+patched_config: ValueError: Vocab size in MultiMolecule RNA alphabet (26) does not match the one in config (28) | AutoModel+manual_state_dict: ValueError: Vocab size in MultiMolecule RNA alphabet (26) does not match the one in config (28) | AutoModelForMaskedLM: ValueError: Vocab size in MultiMolecule RNA alphabet (26) does not match the one in config (28) | AutoModelForMaskedLM+patched_config: ValueError: Vocab size in MultiMolecule RNA alphabet (26) does not match the one in config (28) | AutoModelForMaskedLM+manual_state_dict: ValueError: Vocab size in MultiMolecule RNA alphabet (26) does not match the one in config (28)
```

## caduceus_ps_1k_d118
```text
AutoModel: ImportError: This modeling file requires the following packages that were not found in your environment: mamba_ssm. Run `pip install mamba_ssm` | AutoModel+patched_config: ImportError: This modeling file requires the following packages that were not found in your environment: mamba_ssm. Run `pip install mamba_ssm` | AutoModel+manual_state_dict: FileNotFoundError: manual state_dict fallback requires pytorch_model.bin | AutoModelForMaskedLM: ImportError: This modeling file requires the following packages that were not found in your environment: mamba_ssm. Run `pip install mamba_ssm` | AutoModelForMaskedLM+patched_config: ImportError: This modeling file requires the following packages that were not found in your environment: mamba_ssm. Run `pip install mamba_ssm` | AutoModelForMaskedLM+manual_state_dict: FileNotFoundError: manual state_dict fallback requires pytorch_model.bin
```

## caduceus_ph_1k_d118
```text
AutoModel: ImportError: This modeling file requires the following packages that were not found in your environment: mamba_ssm. Run `pip install mamba_ssm` | AutoModel+patched_config: ImportError: This modeling file requires the following packages that were not found in your environment: mamba_ssm. Run `pip install mamba_ssm` | AutoModel+manual_state_dict: FileNotFoundError: manual state_dict fallback requires pytorch_model.bin | AutoModelForMaskedLM: ImportError: This modeling file requires the following packages that were not found in your environment: mamba_ssm. Run `pip install mamba_ssm` | AutoModelForMaskedLM+patched_config: ImportError: This modeling file requires the following packages that were not found in your environment: mamba_ssm. Run `pip install mamba_ssm` | AutoModelForMaskedLM+manual_state_dict: FileNotFoundError: manual state_dict fallback requires pytorch_model.bin
```

## caduceus_ps_131k_d256
```text
AutoModel: ImportError: This modeling file requires the following packages that were not found in your environment: mamba_ssm. Run `pip install mamba_ssm` | AutoModel+patched_config: ImportError: This modeling file requires the following packages that were not found in your environment: mamba_ssm. Run `pip install mamba_ssm` | AutoModel+manual_state_dict: FileNotFoundError: manual state_dict fallback requires pytorch_model.bin | AutoModelForMaskedLM: ImportError: This modeling file requires the following packages that were not found in your environment: mamba_ssm. Run `pip install mamba_ssm` | AutoModelForMaskedLM+patched_config: ImportError: This modeling file requires the following packages that were not found in your environment: mamba_ssm. Run `pip install mamba_ssm` | AutoModelForMaskedLM+manual_state_dict: FileNotFoundError: manual state_dict fallback requires pytorch_model.bin
```

## carbon_500m
```text
OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'. | OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
```

## evo2_1b_base
```text
ValueError: Couldn't instantiate the backend tokenizer from one of: 
(1) a `tokenizers` library serialization file, 
(2) a slow tokenizer instance to convert or 
(3) an equivalent slow tokenizer class to instantiate and convert. 
You need to have sentencepiece or tiktoken installed to convert a slow tokenizer to a fast one. | ValueError: Couldn't instantiate the backend tokenizer from one of: 
(1) a `tokenizers` library serialization file, 
(2) a slow tokenizer instance to convert or 
(3) an equivalent slow tokenizer class to instantiate and convert. 
You need to have sentencepiece or tiktoken installed to convert a slow tokenizer to a fast one.
```
