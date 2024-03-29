# Learning Engineered Code from Software Quality Metrics

We created this anonymous GitHub account to publish our experimental data for the reviewers. Later we intend to move this repository to our group's account.

Below you can find info about all the files contained in the repository. Basically we devided everything into DATA and SCRIPTS. The scripts are self-explanatory. 

*Important Note*: you need to first unzip processed-X.csv.zip in order to run the scripts properly.

+ data - contains all the data used in our experiments
    - processed-X.csv.zip - zip file containing all metric values (filtered - SLOC>3 - and normalized by NTOKENS)
    - processed-y.csv - contains the labels for each method in X (engineered variable - 0 or 1)
    - raw - contains all the data without preprocessing (raw, without filtering by SLOC and non-normalized)
        * github-10k-non-eng.csv.zip - zip file with data from the non-engineered sample methods
        * github-550-eng.csv.zip - zip file with data from the engineered sample methods
+ scripts - contains all the scripts necessary to run the experiments
    - pre-processing.py - Python script for feature selection and scaling
    - correlation_analysis.py - Python script to perform the correlation analysis
    - ml-script-various-classifiers.py - Python script to train and test all classifiers
    - hyper-parameter-tuning-RF.py - Python script to perform hyper-parameter tuning on Random Forest
    - script.R - R script to obtain mean values for each metric and to run Wilcoxon tests

