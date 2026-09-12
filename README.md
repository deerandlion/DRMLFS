# DRMLFS

This repository provides the source code for the paper:

**DRMLFS: [Dynamic Redundancy Modeling for Multi-Label Feature Selection with Label Supplementation]**

## Overview

This repository contains the implementation of **DRMLFS**, a dynamic redundancy-aware multi-label feature selection method.

The main files are:

* `new.py` — implementation of the proposed DRMLFS method.
* `newalg.py` — main program for running the DRMLFS algorithm.
* `environment.yml` — Conda environment configuration.
* `flag/train/` — training-related data or configuration files.
* `flag/test/` — testing-related data or configuration files.

## Environment

The code was developed and tested in a Python environment managed by Conda.

The required environment can be created using:

```bash
conda env create -f environment.yml
```

After creating the environment, activate it with:

```bash
conda activate <environment-name>
```

Please refer to `environment.yml` for the specific dependencies and package versions.

## Usage

After installing the required dependencies, run the main program:

```bash
python newalg.py
```

The implementation of the proposed method is provided in:

```text
new.py
```

## Data

The `flag/train/` and `flag/test/` directories contain the files required by the current implementation.

Please ensure that the directory structure is maintained when running the code:

```text
DRMLFS/
├── environment.yml
├── new.py
├── newalg.py
└── flags-train.arff/
└── flags-test.arff/
```

## Reproducibility

The provided source code and environment configuration are intended to facilitate the reproduction of the experimental results reported in the paper.

## License

This repository is provided for academic and research purposes.

