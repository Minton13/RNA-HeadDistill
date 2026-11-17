# Model weights: identity, integrity and licences

This file pins the exact third-party checkpoints used in the paper and states
how each one may be obtained. Every weight listed here was verified on
2026-11-16: the SHA256 of the local file is identical to the download-time
LFS etag recorded by the Hugging Face client, so the pair
`(HF repo, revision, SHA256)` unambiguously identifies the bytes we used.

Full machine-readable version:
[`assets/manifests/model_identity_revision_sha256.csv`](assets/manifests/model_identity_revision_sha256.csv).

## Teacher checkpoints and student backbone

| Model (registry name) | Role | Source repository | Revision (commit) | `model.safetensors` SHA256 | Licence | How to obtain |
|---|---|---|---|---|---|---|
| `splicebert` | teacher (L6 bank) | `multimolecule/splicebert` | `c15b162ae5297e3828b6c3c3c8269744cfabfd53` | `64019288d5d5bccbfba81d2b386cabc9c32061623b35b9ce43b66e226ccb3bf4` | AGPL-3.0 (MultiMolecule) | GitHub Release tarball (see below) |
| `plantrna-fm` | teacher | `yangheng/PlantRNA-FM` | `5e7baf4e9d884f7a3ff7385b6a90dd2f63c7b1e5` | `01f66172a243d589945b0b270cde3c5567c09d4ba537fcd710c8295e0d60d8c4` | MIT | GitHub Release tarball, or HF (`main` = same commit) |
| `nucleotide-transformer-v2-500m` | teacher | `InstaDeepAI/nucleotide-transformer-v2-500m` | `06615c1660c892fc199840c18123f8385b3542a8` | `7a00ffaaf8f3ca627741a61e568d7e550b5c79aada70c76c8cb861192fbe7c59` | CC BY-NC-SA 4.0, **HF gated** | download from HF with an accepted access request — **not redistributed here** |
| `rinalmo-mega` | teacher | `multimolecule/rinalmo-mega` | `69e7755c761b23d6a373fd0e76eecad03d5736bc` | `541a2861605a48dc622bf4a2245285f2240771911d79d9abed146eb81d7ffae7` | AGPL-3.0 (MultiMolecule) | GitHub Release tarball |
| `rnafm` | teacher | `multimolecule/rnafm` | `dc0860a3c7d09bf8c4e7c582f47801b6fd8cafbc` | `b3780f7c986be3aef4eb601bbcfae71c63a647a95e6d4f340fe8cc3922e88f7f` | AGPL-3.0 (MultiMolecule) | GitHub Release tarball |
| `utrlm-mrl` | teacher | `multimolecule/utrlm-mrl` | `7ff373a5cbad8c547c67ab254c0dbe51b6acdf60` | `4a7b099bf07740f61d5bd7ce96fdd3df869834879d2c36ac0cc82bb7afd9c8ee` | AGPL-3.0 (MultiMolecule) | GitHub Release tarball |
| `rinalmo-giga` | teacher | `multimolecule/rinalmo-giga` | `188496d6975b57d591baa5634c85cdb7cf5fa891` | `43d77d6b640da560dd1e5cefe749f1dc71b5228f026599cee788a10c693cb780` | AGPL-3.0 (MultiMolecule) | GitHub Release tarball |
| `moderngena-base` | teacher | `AIRI-Institute/moderngena-base` | `7036ebb6eccbd299e7941363a15712706775c64e` | `bc5c2a68cb2b98b2f6903ee1e06ceef3501ccae0307f3f55eb54cb0ae8ea4498` | no licence stated | download from HF (`main` = same commit) — **not redistributed here** |
| `mrnafm` | teacher | `multimolecule/mrnafm` | `955dbc9f425b2aa9ae307f7e3db61f961a6106f5` | `88748e211df78cf636a542848e45e57d04f541da6405981a6c2b2ae3a3050447` | AGPL-3.0 (MultiMolecule) | HF (`main` = same commit), or GitHub Release tarball |
| `rinalmo-micro` | student backbone | `multimolecule/rinalmo-micro` | `74d497a699380928de653366cd7fe1d4da49dae7` | `8519ec91292dc1d31df593e80aa7f980781c11fcebfbbfbdef58249c9855ad12` | AGPL-3.0 (MultiMolecule) | GitHub Release tarball |

## Why tarballs: the 2026-05-31 MultiMolecule repackaging

On or around 2026-05-31 the MultiMolecule project repacked its weight
repositories; the exact commits listed above are **no longer publicly
addressable** upstream for `splicebert`, `rnafm`, `rinalmo-micro/-mega/-giga`
and `utrlm-mrl`. Because those bytes exist only in our archive, we redistribute
the unmodified snapshots (including each repository's original `README.md`,
`license.md` and `license-faq.md`) as a GitHub Release asset, split into
<2 GiB parts:

```bash
# after downloading the part files from the release page:
cat teacher-weights-multimolecule-agpl_v1.0.0.tar.gz.part-* \
    > teacher-weights-multimolecule-agpl_v1.0.0.tar.gz
tar -xzf teacher-weights-multimolecule-agpl_v1.0.0.tar.gz   # creates one dir per model
sha256sum -c SHA256SUMS
```

Redistribution of these weight files is permitted by their AGPL-3.0 terms;
the licence texts are included inside the tarball. `plantrna-fm` (MIT) is
shipped as a separate tarball. `nucleotide-transformer-v2-500m` (gated,
non-commercial) and `moderngena-base` (no licence granted) are **not**
redistributed; fetch them from Hugging Face at the pinned revisions above.

## Trained student checkpoint licence

The released student checkpoint (`stage4b_release.pt` /
`model.safetensors`, GitHub Release asset
`student-rinalmo-micro-head2head_v1.0.0.tar.gz`) was fine-tuned from the
MultiMolecule distribution of RiNALMo-micro. Per the MultiMolecule License FAQ
(`license-faq.md`, Section 2), modified weight files conveyed to third parties
must be conveyed under the same agreement. The student checkpoint is therefore
distributed under **AGPL-3.0**, with the following Corresponding Source
references:

- backbone initialisation: `multimolecule/rinalmo-micro` @ `74d497a6` (table above);
- training pipeline: this repository (MIT);
- distillation corpus: 10% stratified sample of the RNAcentral-based RiNALMo
  pretraining corpus (see the manuscript Methods and
  `assets/manifests/` for provenance; RNAcentral is a public resource);
- supervision signals: outputs of the nine teacher checkpoints listed above.

The alignment adapters and all code in this repository remain under the MIT
licence.
