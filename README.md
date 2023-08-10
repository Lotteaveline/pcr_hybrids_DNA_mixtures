# Denoising tool for hybrids in DNA mixtures from MPS data
The source code for inidicating hybrids and predicting the number of reads per sequence from hybrid formation in MPS DNA mixture data using regression techniques. 

Feature data sets for using in the models can be created in the obtain_hybrid.py by:
```
python3 obtain_hybrids.py "FDSTools output folder", "PCR kit (Mito-mini or Easymito)"
```

There are three models that can be trained, the sklearn RanfomForestREgressor() and GradientBoostingRegressor() and the XGBRegressor() from XGBoost. 

```
python3 machine_learning.py "path_to_obtain_hybrids_output_folder/features_all.py", "path_to_obtain_hybrids_output_folder/labels_all.py", "model_name (random_forest, xgboost_sklearn, xgboost)"
```

And using the models on other data set, such as another test set with unseen individuals:

```
python3 FDSTools_hybrid.py "path_to_obtain_hybrids_output_folder/features_all.py", "path_to_obtain_hybrids_output_folder/labels_all.py", "path_to_model/model"
```

NOTE: FDSTools_hyrbid also contains potential code for the integration of a model in FDSTools. This is not finished, but a start.



Calculating the hybrid potential factor of sequences per marker in a file can be done by:

```
python3 hybrid_potential_factor.py "path_to_sequences_per_marker_file"
```
