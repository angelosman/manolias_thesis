# Script to import predictions from two models and calculate
# Net Reclassification Improvement (NRI)

# Import necessary libraries
import pandas as pd
import sys
from joblib import Parallel, delayed
from tqdm import tqdm
# %%
# Read arguments
path_to_model_1 = sys.argv[1]
path_to_model_2 = sys.argv[2]
output_file = sys.argv[3]

print("Path to model 1 results", path_to_model_1)
print("Path to model 2 results", path_to_model_2)
print("Output file: ", output_file)


# %%


# Crate function
def calculate_NRI(path_1, path_2, bootstrap_sample):

    model_1 = path_1 + 'predictions_lasso_' + str(bootstrap_sample) + '_10folds.csv'
    model_2 = path_2 + 'predictions_lasso_' + str(bootstrap_sample) + '_10folds.csv'

    # Load data
    pred_1 = pd.read_csv(model_1)
    pred_2 = pd.read_csv(model_2)

    # Rename prediction column of pred_1 as original prediction
    pred_1 = pred_1.rename(columns={'prediction': 'prediction_original'})

    # Keep columns event, time, prediction_original and ID
    pred_1 = pred_1[['ID', 'event', 'time', 'prediction_original']]

    # Rename prediction column of pred_2 as new prediction
    pred_2 = pred_2.rename(columns={'prediction': 'prediction_new'})

    # Keep columns event, time, prediction_new and ID
    pred_2 = pred_2[['ID', 'event', 'time', 'prediction_new']]

    # Merge predictions on ID and event columns
    pred_data = pd.merge(pred_1, pred_2, on=['ID', 'event', 'time'])

    # Calculate NRI
    cu_events = sum((pred_data['prediction_new'] > pred_data['prediction_original']) & (pred_data['event'] == 1))
    id_events = sum((pred_data['prediction_new'] < pred_data['prediction_original']) & (pred_data['event'] == 1))

    cd_nonevents = sum((pred_data['prediction_new'] < pred_data['prediction_original']) & (pred_data['event'] == 0))
    iu_nonevents = sum((pred_data['prediction_new'] > pred_data['prediction_original']) & (pred_data['event'] == 0))

    total_events = sum(pred_data['event'] == 1)
    total_nonevents = sum(pred_data['event'] == 0)

    nri_calc = (((cu_events / total_events) - (id_events / total_events)) - (
                (cd_nonevents / total_nonevents) - (iu_nonevents / total_nonevents)))

    return nri_calc
# %%
reff = Parallel(n_jobs=9)(delayed(calculate_NRI)(path_1=path_to_model_1,
                                                  path_2=path_to_model_2,
                                                  bootstrap_sample=i) for i in tqdm(range(1, 201),
                                                                                      total=len(range(1, 201))))
nri_all = pd.DataFrame(reff)

nri_all.to_csv(output_file, index=False)
# %%
