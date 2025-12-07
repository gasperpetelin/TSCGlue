# AutoTSC

- Does selection models based on val performance even work?
- Which ensemble method should AutoML use as a function of dataset size? (stacking double stacking weights like now?)
- Does the validation-test disconnect generalize across domains, or is it TSC-specific?
- Does nested CV actually solve the small-sample problem?
- Does downsampling work?
- Does multifidelity work?
- No AutoML due to small datasets?
- How often is best val model best on train (shuffled/no shuffled)
- Resampling should be done or not?


✅ 1. First-order & local-transform views

These change the shape or local structure of the series.

✔ Differencing (Δx)

Good for removing trends, enhancing sharp transitions.

✔ Cumulative sum

Smooths noise; emphasises long-term structure.

✔ Moving average / smoothing (SG filter, EMA)

Suppresses high-frequency noise → different model inductive bias.

✔ Trend removal (detrending)

Removes global shape and highlights wiggles.

✅ 2. Frequency & phase transforms

Often extremely useful because they create orthogonal representations.

✔ FFT magnitude

Spectral view of the series — deep models love it.

✔ FFT phase

Adds complementary structure to magnitude.

✔ STFT / Sliding FFT

Time-frequency representation (use 1D vector summary or window-level stats).

✔ Wavelet transform (CWT, DWT)

Great for multi-resolution patterns.

✔ Hilbert transform

Creates an analytic signal: amplitude envelope + instantaneous phase.

✅ 3. Shape & geometric transforms

Excellent for diversity because they are structurally different.

✔ Time warping (random or deterministic)

Warps the timeline → great for shape-based algorithms.

✔ Curve length transform

Turns a series into cumulative path length.

✔ Polar coordinate transform

Convert (x, diff(x)) into polar angle + magnitude.

✔ Slope transform

Use local slope or angle instead of raw values.

✅ 4. Normalization-based transforms

Don’t underestimate these — especially helpful with ROCKET/Hydra ensembles.

✔ Z-normalization (per series)

Baseline for most TSC but provides a different view from raw scale.

✔ Min–max scaling

Good for emphasising relative shape.

✔ Unit energy / L2 normalization

Highlights relative oscillations.

✔ Robust scaling (median/IQR)

When outliers distort structure.

✅ 5. Feature extraction transforms

These produce feature vectors that a classifier sees differently from the raw TS.

✔ Catch22

22 interpretable features — often complementary to ROCKET.

✔ TSFresh / TSFEL feature subsets

Huge diversity if you select subsets.

✔ Autocorrelation / partial autocorrelation vectors

Very different inductive bias.

✔ Shapelet distances

Distance to “prototype” shapes.

✅ 6. Windowing & multi-resolution views

Often extremely strong for ensembling.

✔ Multi-scale segment averaging

Compute downsampled versions at multiple resolutions.

✔ Piecewise transforms

PAA (Piecewise aggregate approximation)

PLA (Piecewise linear approximation)

SAX (Symbolic Aggregate Approximation)

✔ Moving-window statistics

Rolling min/max/std/skew/kurt.

✅ 7. Noise & augmentation transforms (for diversity)

Used often in Hydra/ROCKET ensembles to create diverse models.

✔ Add small Gaussian noise

Mild regularization; different learned filters.

✔ Jittering / scaling

Preserves topology but shifts amplitude.

✔ Dropout segments

Removes random subsequences → encourages robustness.