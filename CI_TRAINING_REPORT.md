# Automatic train-only fine-tuning results

Completed encoder sources: 18; current run: 5; carried forward: 13.

Best observed validation result (reporting only, not checkpoint selection): PhoBERT_lan24 **0.898383**.

- PhoBERT_lan11: validation ABSA-F1 0.808473.
- PhoBERT_lan12: validation ABSA-F1 0.798199.
- PhoBERT_lan13: validation ABSA-F1 0.761572.
- PhoBERT_lan14: validation ABSA-F1 0.854359.
- PhoBERT_lan15: validation ABSA-F1 0.781356.
- PhoBERT_lan16: validation ABSA-F1 0.843049.
- PhoBERT_lan17: validation ABSA-F1 0.815299.
- PhoBERT_lan18: validation ABSA-F1 0.784918.
- PhoBERT_lan19: validation ABSA-F1 0.879531.
- PhoBERT_lan20: validation ABSA-F1 0.890595.
- PhoBERT_lan21: validation ABSA-F1 0.508194.
- PhoBERT_lan22: validation ABSA-F1 0.850615.
- PhoBERT_lan23: validation ABSA-F1 0.887253.
- PhoBERT_lan24: validation ABSA-F1 0.898383.
- PhoBERT_lan25: validation ABSA-F1 0.893641.
- PhoBERT_lan26: validation ABSA-F1 0.888622.
- PhoBERT_lan27: validation ABSA-F1 0.887139.
- ViSoBERT_lan2: validation ABSA-F1 0.848298.
- Stacking_lan4: validation ABSA-F1 0.885298.
- Blending_lan3: validation ABSA-F1 0.893641.

Validation has been seen during development. No blind data was opened. Sources selected checkpoints on tune; meta CV is not fully nested encoder CV.

For an outer80 refit, tune probabilities and recipe-selection F1 come from the original pre-refit model, never the refitted model predicting its own training rows. Validation scores come from the refit model; their probability distribution may shift.

Original FP32 checkpoints are cached by this workflow for retrieval; the selected encoder is packaged separately and re-evaluated after FP16 storage conversion.
