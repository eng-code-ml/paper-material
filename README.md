# SANER 2020 - Learning Engineered Code from Software Quality Metrics (experimental material repository)

We created this anonymous GitHub account to publish our experimental data for the reviewers. Later we intend to move this repository to our group's account.

Below you can find info about all the files contained in the repository.

+ data - contains all the data used in our experiments
    - processed-X.csv.zip - zip file containing all metric values (filtered - SLOC>3 - and normalized by NTOKENS)
    - processed-y.csv - contains the labels for each method (engineered variable)
    - raw - contains all the data without preprocessing (raw, without filtering by SLOC and non-normalized)
        * github-10k-non-eng.csv.zip - zip file with data from the non-engineered sample methods
        * github-550-eng.csv.zip - zip file with data from the engineered sample methods
+ scripts - contains all the scripts necessary to run the experiments
    - ml-script-various-classifiers.py - Python script to train and test all classifiers
    - pre-processing.py - Python script for feature selection and scaling
    - script.R - R script to obtain mean values for each metric and to run Wilcoxon tests

