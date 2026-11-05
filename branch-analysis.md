# UCFSHT-specific XD bridge

Preserved branch: `model/UCFSHT`, commit `b68d216d4ee659afb7e087947f1b8e56fa86b330`.

Only `xd_experiment/branch.py` is the branch-specific integration. Existing tracked source files are not modified. Five source files' Git blob hashes are checked at runtime (CRLF checkout conversion alone is tolerated). Model, loss, pseudo and ITA implementations remain independent of APWCS.

- Model: original `Model_V2_AllCNN`, with its accepted `n_features` argument set to actual RGB feature D=1024. Convolutions, gating, pooling and dropout are unchanged.
- Loss/train: original `concatenated_train_variable_length` and its masked BCE/Huber hybrid. Confidence refinement is commented out in the original function; `generate_confidences()` provides float32 ones with a metadata note that these values do not affect the original loss. This is different from APWCS, whose active refinement must remain independent.
- Input: common preparation averages five **real** RGB crop feature arrays to `(T,1024)`. A singleton-axis identity view `(T,1,1024)` permits the original pseudo generator to apply its existing `mean(axis=1)`. This does not create fake crops or features and does not change dimension.
- Pseudo: original `pseudo_oc.py` function with its **main call's** settings: standard feature normalization, zscore scores, initial-five prototype (`prototype_method='none'` chooses original else branch), Gaussian p-value branch (0.1 internally), opening/closing, two-statistic rank weighting and box expansion; prototype swap disabled. Legacy attraction/isolated flags are passed exactly as main but the source has those operations commented out, so they are not described as active contributions. Generator diagnostic plot and metadata go to the requested output folder.
- Short videos: original generator emits zeros for scores when T<8 and fills missing rank statistics from the other videos. The full XD train has sufficient longer videos. An all-short subset is rejected explicitly instead of inventing an alternative rank algorithm.
- Adapter: original `ResidualAdapter2048` with `d=1024`, LayerNorm enabled. Historical class name is not its dimensionality constraint.
- ITA: original `_split_prefix_suffix_video` and `_tta_update_one_video` function bodies/signatures loaded through AST to avoid module-level argparse and UCF data imports. Caller supplies k/q/min_keep/lr/steps explicitly; prefix-only adaptation, detector frozen and LN-only updates remain unchanged. Debug records are available as `adapter._xd_last_adaptation_debug`.

Known legacy ITA semantics are deliberately retained: selected `(N,D)` is interpreted by model as batch N/time1; `(N,1)` losses multiply `(N,)` weights by broadcasting. Margin and LN regularization are calculated but not used by the actual loss. This bridge does **not** silently fix these algorithm behaviors. Any corrected ITA must be a separately labeled experiment.

No `test.py` standalone full/suffix metric code or test-based checkpoint selection is reused: the common XD runner supplies one explicit frame protocol and fixed-epoch training for both branches. The bridge itself never reads test GT or true video labels.

## Bridge CPU check (2026-11-05)

Passed using the existing local Python 3.12 / PyTorch CPU environment:

- Verify all five source blob hashes against preserved commit.
- Load five genuine crops of the shortest real XD train video `v=Gm73TwtUyGY__#1_label_G-0-0`; crop feature mean `(2,1024)` enters the original model as `(1,2,1024)` and produces finite `(1,2,1)` outputs.
- One original training-function step on that two-snippet feature input gives finite loss and supports the genuine shortest-video shape.
- Synthetic pseudo run includes T=2/8/12 videos and returns 22 finite binary float32 targets. Original diagnostics plot and metadata write to isolated output directory.
- Inactive-confidence adapter returns float32 ones with matching timeline length.
- One unchanged original ITA step on `(12,1024)` produces finite state, updates LN only and keeps delta frozen; debug has one step.
- Source hash verification still passes after these checks.

These are tensor/code integration checks, **not XD benchmark performance**. Full GPU training, full XD pseudo generation and final evaluation remain school-PC work.
