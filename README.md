# Identifying profiles, trajectories, burden, social and biological factors in 3.3 million individuals with multimorbidity in England

This repository houses all the python scripts utilized in both multimorbidity profile identification and clinical marker relevance identification as detailed in our study.

## Background
Multimorbidity, the co-occurrence of multiple chronic conditions in an individual, has become a global health challenge affecting populations in high-income and low- to middle-income countries. Despite its increasing prevalence, critical gaps remain in understanding its progression, burden, and determinants to better guide prevention and treatment. Here, by leveraging linked primary care, hospitalisation, and mortality records from 3.3 million individuals with multimorbidity in England, we conducted a longitudinal cohort study to characterise multimorbidity across multiple dimensions, including condition profiling, progression trajectories, healthcare burden, and social and biological factors.

## System Requirements

### Running Environment

The Python scripts in this repository are designed to run within a Python environment. Key requirements include:

- Python Version: 3.8.18.
- GPU and RAM Configuration: This code was developed and tested on a workstation with an NVIDIA RTX A5500 GPU (24GB VRAM) and 256GB system RAM.

### Python Packages
- stepmix: 2.2.1
- shap: 0.44.1
- xgboost: 2.1.1
- pandas: 1.5.3
- scipy: 1.10.1

## Usage
### Compute Multimorbidity Clusters Using Latent Class Analysis (LCA) Method
Example input data for the LCA method is available in the data/cond_vector directory, containing binary condition vectors where 1 indicates presence and 0 indicates absence of chronic conditions for individuals within specific age-sex groups. Execute the LCA analysis using:
```python
python 1_run_lca.py
```
This script:
- Tests cluster solutions from 2 to 15 profiles by default
- Processes data separately for each age band and sex group
- Saves individual cluster assignments (i.e., individual-cluster mapping) and goodness-of-fit statistics (BIC, AIC, CAIC) in the results folder

Generate score plots to assess optimal cluster numbers with:
```python
python 2_cluster_number.py
```
The plots visualize goodness-of-fit metrics across different cluster solutions to help determine the most appropriate number of profiles. The analysis identifies the optimal number of clusters for each age-sex combination and stores this information in the `age_gender2nk` dictionary within `3_hierarchical_profile.py` for use in subsequent hierarchical clustering.

### Identify Multimorbidity Profiles Using Hierarchical Clustering Method
To create unified multimorbidity profiles across age bands, execute:
```python
python 3_hierarchical_profile.py
```
This script:
- Combines clusters identified in different age bands through hierarchical clustering
- Generates comprehensive profile definitions that span the life course

### Trajectory, Burden and Social Factors
As we have mapped each individual to a multimorbidity profile in each age band, we can now derive the following:
- **Trajectory**: An individual’s multimorbidity trajectory is defined by the sequence of profiles to which they belong across successive age bands.
- **Burden**: For each profile and age band, we calculate three burden metrics of mortality rate, hospitalisation rate and hospitalisation prevalence, using the corresponding outcome data of all individuals assigned to that profile.
- **Social Factors**: By stratifying the cohort by key social determinants (e.g. socioeconomic deprivation, ethnicity, geographic region), we compute the prevalence of each multimorbidity profile within each subgroup.

### Identify Relevant Markers to Each Multimorbidity Profile
The example data of input for the interpretable machine learning framework can be found within `data/marker` folder. Execute the full analytical pipeline with:
```python
python 4_xgboost_shap.py
```
This script:
- Splits the train and test sets
- Trains and evaluates XGBoost model
- Conducts reference-adjusted SHAP value calculation for clinical relevance

The relevance of each marker for each profile is output and stored in `results/relevance_marker_prof_M.csv` or `results/relevance_marker_prof_F.csv` for males and females, respectively.