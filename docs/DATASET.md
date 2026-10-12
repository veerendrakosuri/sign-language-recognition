# WLASL data preparation

Use the [official WLASL repository](https://github.com/dxli94/WLASL) as the source of truth for metadata, access instructions, the paper, and dataset terms. Read its C-UDA agreement before using the dataset. This project does not accept terms on your behalf or redistribute the videos.

1. Obtain `WLASL_v0.3.json` from the upstream `start_kit` directory, retaining original ordering and split fields.
2. Follow the upstream video access workflow. Its downloader retrieves original videos; its preprocessing step creates annotated sign clips. The project also describes requesting preprocessed/missing videos.
3. Put preprocessed clips in `data/videos/<video_id>.mp4` and metadata at `data/WLASL_v0.3.json`.
4. Run the manifest command and inspect coverage before extracting landmarks.

Optional way to inspect upstream material locally:

```powershell
git clone https://github.com/dxli94/WLASL.git data/WLASL-source
```

Then read that checkout's instructions and terms. The repository here does not blindly execute the upstream downloader, request missing videos, or download a third-party mirror. If you use a mirror, verify provenance, terms, clip boundaries, labels and official splits; do not trust the mirror's train/test folder names.

## Trimming matters

This extractor expects **already trimmed** videos and deliberately does not apply metadata frame ranges again. Upstream ranges refer to original videos and their stated decoding rate; double trimming a preprocessed clip destroys examples. When processing original sources yourself, follow the upstream preprocessing implementation and its frame-rate convention. Preserve the full sign, including movement at the start and end.

## Subsets and missing files

The first K glosses in the official metadata define the subset. Use K=100 for WLASL100. Do not recompute popularity from downloaded files, sort alphabetically, or reshuffle instances. `split` is preserved as train/val/test and `signer_id` retained for audits. The code does not claim a new signer-disjoint evaluation beyond the upstream split definition.

The manifest records expected and available files; extraction additionally records unusable clips. A file's presence is not proof that its labels or trimming are correct. Manually review examples. Evaluation records exact used IDs, missing classes, checkpoint hash and denominator. Missing data must be disclosed alongside results.

Metadata, videos, caches, and detector assets are ignored by Git. Keep a local copy of your metadata and file hashes to reproduce a particular experiment.
