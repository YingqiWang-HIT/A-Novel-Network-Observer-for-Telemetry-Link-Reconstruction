# Baseline adapters

This folder contains **one independent `.py` file per comparison method**. These files are public-release adapters/placeholders only.

For copyright and authorship reasons, this repository does **not** redistribute the official implementations of third-party comparison methods, including Kalman smoother, SAITS, GRIN, SPIN, CSDI, BayOTIDE, Mamba, LinOSS, TIMBA, KEDGN, HyperIMTS, and ImDiffusion. Please read the original papers carefully and obtain code from the original authors' official repositories or release channels.

Recommended workflow:

1. Download or clone the official baseline implementation from the original authors.
2. Put it under `baselines/external/<BaselineName>/`.
3. Complete the corresponding adapter file in this folder so it exposes a consistent `fit()` / `predict()` interface.
4. Report clearly whether results are obtained from official code, reimplementation, or an adapter.

Do not copy third-party code into this repository unless its license explicitly permits redistribution and the license notice is preserved.
