# A Multi-atlas Ensemble Learning Framework for Autism Spectrum Disorder Diagnosis
## Environment Setup

Please be aware that this code is meant to be run with Python 3 under Linux. Download the packages from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```
We use the [ABIDE I dataset](http://fcon_1000.projects.nitrc.org/indi/abide/). We use the pre-processing method from the [preprocessed-connectomes-project](https://github.com/preprocessed-connectomes-project/abide).Due to issues such as missing time series, the incomplete brain coverage, and severe head motion artifacts, some data samples are excluded from this study. Ultimately, valid data from 403 individuals with ASD (positive class) and 468 TC (negative class) are included, totaling 871 subjects.
## Steps for running the experiment

1. Run `download_abide.py` to download the raw data.
Use the command below:

```bash
#download_abide.py [--pipeline=cpac] [--strategy=filt_global] [<derivative> ...]
python download_abide.py
```
