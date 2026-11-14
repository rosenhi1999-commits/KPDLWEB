# Automatic train-only fine-tuning results

Completed encoder trials: 8 / 8.

- PhoBERT_lan11: validation ABSA-F1 0.808473.
- PhoBERT_lan12: validation ABSA-F1 0.798199.
- PhoBERT_lan13: validation ABSA-F1 0.761572.
- PhoBERT_lan14: validation ABSA-F1 0.854359.
- PhoBERT_lan15: validation ABSA-F1 0.781356.
- PhoBERT_lan16: validation ABSA-F1 0.843049.
- PhoBERT_lan17: validation ABSA-F1 0.815299.
- PhoBERT_lan18: validation ABSA-F1 0.784918.
- Stacking_lan2: validation ABSA-F1 0.860469.

Validation has been seen during development. No blind data was opened. Sources selected checkpoints on tune; meta CV is not fully nested encoder CV.

Original FP32 checkpoints are cached by this workflow for retrieval; the selected encoder is packaged separately and re-evaluated after FP16 storage conversion.
