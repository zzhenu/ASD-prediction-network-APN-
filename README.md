# A Multi-atlas Ensemble Learning Framework for Autism Spectrum Disorder Diagnosis
## Environment Setup

Please be aware that this code is meant to be run with Python 3 under Linux. Download the packages from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```
We use the [ABIDE I dataset](http://fcon_1000.projects.nitrc.org/indi/abide/). We use the pre-processing method from the [preprocessed-connectomes-project](https://github.com/preprocessed-connectomes-project/abide).Due to issues such as missing time series, the incomplete brain coverage, and severe head motion artifacts, some data samples are excluded from this study. Ultimately, valid data from 403 individuals with ASD (positive class) and 468 TC (negative class) are included, totaling 871 subjects.
## Steps for running the experiment

1. Download the raw data.
Use the command below:

```bash
python download_abide.py
```
2. Display the data distribution

```bash
python ABIDE_analysis.py
```
3. Run `new_prepare_data.py` to compute correlations and obtain HDF5 files.
```bash
python new_prepare_data.py --folds=10 --whole cc400 aal dosenbach160
```
4. Using VAEA to perform Multi-atlas Deep Feature Representation, and using multilayer perceptron (MLP) and ensemble learning to classify the ASD and TC.
```bash
python new_vae_ae_many_nn.py --whole cc400 aal dosenbach160
```
5. Evaluating the MLP model on test dataset.Evaluate the MLP model on the test dataset. You can use the models previously saved in the "best" folder, or directly run this command.
```bash
python new_esemble_many_871.py --whole  aal dosenbach160 cc400
```


