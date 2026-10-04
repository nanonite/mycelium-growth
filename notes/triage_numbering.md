# Triage: Sequential File Numbering vs Time/Dish (#22)

**Question.** Does the sequential file numbering (00000001–00020176) encode capture time, dish identity, or both?

## Summary

| Archive | Images | ID Range | Fwd steps (ID→mtime) | Fwd steps (ID→bright) | Bright autocorr@1 |
|---|---|---|---|---|---|
| labeled-GL.zip | 507 | 1–507 | 0.5474 | 0.5474 | 0.0311 |
| labeled-GS_PO_TS.zip | 30 | 18428–18457 | 0.8966 | 0.5517 | -0.0748 |
| labeled-MYG_PDA_TEMP.zip | 30 | 18458–18487 | 0.7931 | 0.4138 | 0.0567 |
| unlabeled-GL1.zip | 4480 | 508–4987 | 0.5015 | 0.5376 | -0.0102 |
| unlabeled-GL2.zip | 2240 | 4988–7227 | 0.5083 | 0.5377 | -0.0116 |
| unlabeled-GL3.zip | 2240 | 7228–9467 | 0.4891 | 0.5328 | -0.0230 |
| unlabeled-GL4.zip | 2240 | 9468–11707 | 0.4962 | 0.5449 | -0.0245 |
| unlabeled-GL5.zip | 2240 | 11708–13947 | 0.4993 | 0.5395 | 0.0045 |
| unlabeled-GL6.zip | 2240 | 13948–16187 | 0.5047 | 0.5364 | 0.0142 |
| unlabeled-GL7.zip | 2240 | 16188–18427 | 0.5056 | 0.5346 | -0.0217 |
| unlabeled-GS_PO_TS.zip | 1689 | 18488–20176 | 0.5012 | 0.5930 | 0.0511 |

## Detailed Analysis

### labeled-GL.zip

- **Images**: 507
- **ID range**: 1–507 (gaps: 0)
- **Forward steps (ID → mtime)**: 0.5474
- **Forward steps (ID → bright_frac)**: 0.5474
- **Run lengths of ID diffs**: [506]
- **Max run length**: 506
- **Bright fraction mean ± std**: 0.0097 ± 0.0260
- **Bright autocorrelation**: {1: 0.03109245858007697, 2: -0.042740771358325616, 5: 0.07024374908723356, 10: 0.060363885991390365, 20: 0.005168849293002308, 50: 0.06860281958454091, 100: 0.0148020015005246, 200: 0.032034647057875794, 500: 0.1729226157171815}

### labeled-GS_PO_TS.zip

- **Images**: 30
- **ID range**: 18428–18457 (gaps: 0)
- **Forward steps (ID → mtime)**: 0.8966
- **Forward steps (ID → bright_frac)**: 0.5517
- **Run lengths of ID diffs**: [29]
- **Max run length**: 29
- **Bright fraction mean ± std**: 0.0018 ± 0.0028
- **Bright autocorrelation**: {1: -0.07476385694559136, 2: 0.025728788725641966, 5: 0.03949680874586146, 10: -0.09253365800349421, 20: 0.3068682524792281}

### labeled-MYG_PDA_TEMP.zip

- **Images**: 30
- **ID range**: 18458–18487 (gaps: 0)
- **Forward steps (ID → mtime)**: 0.7931
- **Forward steps (ID → bright_frac)**: 0.4138
- **Run lengths of ID diffs**: [29]
- **Max run length**: 29
- **Bright fraction mean ± std**: 0.0064 ± 0.0119
- **Bright autocorrelation**: {1: 0.05668775838709678, 2: 0.04871894701515665, 5: 0.37046996689037665, 10: -0.0580199641094397, 20: 0.46417860656924237}

### unlabeled-GL1.zip

- **Images**: 4480
- **ID range**: 508–4987 (gaps: 0)
- **Forward steps (ID → mtime)**: 0.5015
- **Forward steps (ID → bright_frac)**: 0.5376
- **Run lengths of ID diffs**: [4479]
- **Max run length**: 4479
- **Bright fraction mean ± std**: 0.0598 ± 0.1314
- **Bright autocorrelation**: {1: -0.010240016156233182, 2: 0.005703347832029868, 5: -0.0052039264944647974, 10: 0.006061308407647498, 20: -0.01622184712573385, 50: 0.003402194020121345, 100: -0.024007310829840412, 200: -0.025088328999531666, 500: -0.00037457481801204826}

### unlabeled-GL2.zip

- **Images**: 2240
- **ID range**: 4988–7227 (gaps: 0)
- **Forward steps (ID → mtime)**: 0.5083
- **Forward steps (ID → bright_frac)**: 0.5377
- **Run lengths of ID diffs**: [2239]
- **Max run length**: 2239
- **Bright fraction mean ± std**: 0.0564 ± 0.1247
- **Bright autocorrelation**: {1: -0.011555037149007883, 2: -0.011636097955930647, 5: -0.010606381755744139, 10: 0.018231478126562627, 20: 0.015956513122483133, 50: -0.017422010531179595, 100: -0.02912494910456646, 200: 0.015208053076609376, 500: -0.048107209932276525}

### unlabeled-GL3.zip

- **Images**: 2240
- **ID range**: 7228–9467 (gaps: 0)
- **Forward steps (ID → mtime)**: 0.4891
- **Forward steps (ID → bright_frac)**: 0.5328
- **Run lengths of ID diffs**: [2239]
- **Max run length**: 2239
- **Bright fraction mean ± std**: 0.0609 ± 0.1290
- **Bright autocorrelation**: {1: -0.02301383605152176, 2: -0.045479617424983514, 5: 0.027238422096575413, 10: 0.009746930357376438, 20: -0.011176695636584617, 50: 0.03211864211775002, 100: 0.03809511870782431, 200: 0.01505003017789873, 500: 0.040978742749157124}

### unlabeled-GL4.zip

- **Images**: 2240
- **ID range**: 9468–11707 (gaps: 0)
- **Forward steps (ID → mtime)**: 0.4962
- **Forward steps (ID → bright_frac)**: 0.5449
- **Run lengths of ID diffs**: [2239]
- **Max run length**: 2239
- **Bright fraction mean ± std**: 0.0650 ± 0.1415
- **Bright autocorrelation**: {1: -0.024500468856744877, 2: 0.02973931745418669, 5: -0.01192757800275122, 10: -0.03529145746525259, 20: 0.011731593679698353, 50: 0.012443823161638754, 100: -0.004240616347641851, 200: -0.03161960611448272, 500: 0.003697735799098169}

### unlabeled-GL5.zip

- **Images**: 2240
- **ID range**: 11708–13947 (gaps: 0)
- **Forward steps (ID → mtime)**: 0.4993
- **Forward steps (ID → bright_frac)**: 0.5395
- **Run lengths of ID diffs**: [2239]
- **Max run length**: 2239
- **Bright fraction mean ± std**: 0.0590 ± 0.1285
- **Bright autocorrelation**: {1: 0.004527425316708876, 2: -0.03530652810615172, 5: -0.006163214590423115, 10: 0.005274354836861792, 20: 0.01996972128514598, 50: -0.006214495260746895, 100: 0.012429618603548654, 200: -0.02909970232598762, 500: -0.0036883610667910295}

### unlabeled-GL6.zip

- **Images**: 2240
- **ID range**: 13948–16187 (gaps: 0)
- **Forward steps (ID → mtime)**: 0.5047
- **Forward steps (ID → bright_frac)**: 0.5364
- **Run lengths of ID diffs**: [2239]
- **Max run length**: 2239
- **Bright fraction mean ± std**: 0.0619 ± 0.1338
- **Bright autocorrelation**: {1: 0.014247812771925699, 2: -0.004838031297329647, 5: 0.027603425239575267, 10: -0.008669953262287816, 20: -0.017838156637479688, 50: 0.03546267178588431, 100: 0.006278201966474312, 200: 0.02510197256463413, 500: -0.025339024422009356}

### unlabeled-GL7.zip

- **Images**: 2240
- **ID range**: 16188–18427 (gaps: 0)
- **Forward steps (ID → mtime)**: 0.5056
- **Forward steps (ID → bright_frac)**: 0.5346
- **Run lengths of ID diffs**: [2239]
- **Max run length**: 2239
- **Bright fraction mean ± std**: 0.0600 ± 0.1315
- **Bright autocorrelation**: {1: -0.021694805007330658, 2: 0.012392139023077509, 5: -0.015303585067904774, 10: 0.049992278664774514, 20: 0.009736630240634352, 50: -0.011345592474715576, 100: -0.011714522408574634, 200: 0.015667221151767597, 500: 0.012192249721822026}

### unlabeled-GS_PO_TS.zip

- **Images**: 1689
- **ID range**: 18488–20176 (gaps: 0)
- **Forward steps (ID → mtime)**: 0.5012
- **Forward steps (ID → bright_frac)**: 0.5930
- **Run lengths of ID diffs**: [1688]
- **Max run length**: 1688
- **Bright fraction mean ± std**: 0.0055 ± 0.0171
- **Bright autocorrelation**: {1: 0.051139267448862546, 2: 0.01577898989379864, 5: 0.018369606490249362, 10: -0.017952516030747354, 20: -0.00042649707853383766, 50: -0.006774488346563216, 100: -0.005216758777605643, 200: -0.019330583627077957, 500: -0.004634299826056196}

## Verdict

**Numbering encodes: NEITHER**

The sequential file numbering is shuffled relative to both export time (zip mtime) and growth progression (bright fraction). No ordering signal recovered.
