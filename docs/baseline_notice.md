# Notice on comparison methods

The comparison methods used in the paper are independent works by their respective authors. To respect the original authors' rights, this repository does not redistribute their official code or any modified copy of it.

For public release, each baseline has a separate adapter file under `baselines/`. The adapter documents the intended integration point but raises an explicit error until the user installs the official implementation locally.

When reproducing results, please:

1. read the original paper of each comparison method;
2. check the official license and citation requirement;
3. obtain the official implementation from the authors' release channel;
4. place it under `baselines/external/<BaselineName>/` or another local path not committed to this repository;
5. record the commit hash, version, hyperparameters, and any local modifications.

This policy avoids accidentally publishing third-party code without permission while keeping the PILOT training/evaluation pipeline transparent.
