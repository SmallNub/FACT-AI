# FACT-AI
This repository hosts the code for "Reproducing MORAL: Fairness in Link Prediction". This repository requires a proper conda environment. The conda evironment file can be found in `environment.yml`, which can be installed using conda:

```bash
conda env create -f environment.yml
conda activate FACT
```

Additionally, a cuda GPU is required to run the training code. If results are already computed, only the evaluation code can be run to bypass the long training times.

The notebook that runs everything is in `MORAL/run_all.ipynb`.
