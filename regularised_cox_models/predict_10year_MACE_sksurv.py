# Script to predict risk of MACE in 10 years using proteomics data.
# Import libraries
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from joblib import Parallel, delayed
from tqdm import tqdm
import joblib
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis as CoxPHFitter
from sksurv.metrics import concordance_index_censored
from sklearn.compose import ColumnTransformer
from sklearn.utils import resample
#%%
# Define path to directory to save results
input_data = sys.argv[1]
path_to_results = sys.argv[2]
no_folds = sys.argv[3]
no_folds = int(no_folds)
random_seed = sys.argv[4]
random_seed = int(random_seed)
outcome = sys.argv[5]
demo = sys.argv[6].lower() == 'true'
clin = sys.argv[7].lower() == 'true'
med = sys.argv[8].lower() == 'true'
CL = sys.argv[9].lower() == 'true'
frs = sys.argv[10].lower() == 'true'
top_prot = sys.argv[11].lower() == 'true'


print("Path to input data: ", input_data)
print("Path to save results: ", path_to_results)
print("Number of folds used: ", no_folds)
print("Random seed selected: ", random_seed)
#%%
# Read data
data = pd.read_csv(input_data)
matched_data = pd.read_csv('/rds/general/user/am6220/projects/advance-omics/live/MACE_results/data'
                           '/matched_data_for_injury.csv')

# Drop participants in matched_data from data
data = data[~data['ID'].isin(matched_data['ID'])]
#%%
# If top_prot is True load amp_signature
amp_sign = pd.read_csv('/rds/general/user/am6220/projects/advance-omics/live/machine_learning/ML_class_injury'
                       '/injury_status_Lasso/beta_prot_models_all.csv')

# Select column names of data that contain "prot"
prot_columns = data.columns[data.columns.str.contains("prot")]

# Select the character between the first and second underscore
prot_columns = prot_columns.str.split('_').str[1]

# Filter for model == "amp"
amp_sign = amp_sign[amp_sign['model'] == "amp"]

# Filter validation_class == "Gold"
amp_sign = amp_sign[amp_sign['validation_class'] == "Gold"]

# arrange by descending order of score
amp_sign = amp_sign.sort_values(by='score', ascending=False)

# Keep UniProt ID in prot_columns
amp_sign = amp_sign[amp_sign['UniProt'].isin(prot_columns)]

# Select top 21 proteins
amp_sign = amp_sign.head(21)
#%%
# Drop columns amputation, serious_accident, combat
data = data.drop(columns=['amputation', 'serious_accident', 'combat'])
#%%
# if outcome is MACE drop columns for MI, HF, stroke and CVDeath
if outcome == 'MACE':
    data = data.drop(columns=['MI', 'HF', 'stroke', 'CVDeath'])

if outcome == 'MI':
    data = data.drop(columns=['event', 'HF', 'stroke', 'CVDeath'])
    # rename MI column to event
    data.rename(columns={'MI': 'event'}, inplace=True)

if outcome == 'HF':
    data = data.drop(columns=['event', 'MI', 'stroke', 'CVDeath'])
    # rename HF column to event
    data.rename(columns={'HF': 'event'}, inplace=True)

if outcome == 'stroke':
    data = data.drop(columns=['event', 'MI', 'HF', 'CVDeath'])
    # rename stroke column to event
    data.rename(columns={'stroke': 'event'}, inplace=True)

if outcome == 'CVDeath':
    data = data.drop(columns=['event', 'MI', 'HF', 'stroke'])
    # rename CVDeath column to event
    data.rename(columns={'CVDeath': 'event'}, inplace=True)
#%%
cols_2_keep = ['ID', 'time', 'event']

# If FRS == True, add columns ending in "FRS" to cols_2_keep
if frs == True:
    cols_2_keep.extend(data.columns[data.columns.str.endswith("FRS")])

if frs == False:
    # If demo/clin is True, add columns starting with "demo"/"clin" to cols_2_keep
    if demo == True:
        cols_2_keep.extend(data.columns[data.columns.str.startswith("demo")])

    if clin == True:
        cols_2_keep.extend(data.columns[data.columns.str.startswith("clin")])

# if med is True, add columns containing "med" to cols_2_keep
if med == True:
    cols_2_keep.extend(data.columns[data.columns.str.contains("med")])

# if CL is True, add columns containing "CL" to cols_2_keep
if CL == True:
    cols_2_keep.extend(data.columns[data.columns.str.contains("CL")])

# Keep unique columns in cols_2_keep
cols_2_keep = list(set(cols_2_keep))

# Keep only the columns in cols_2_keep
data = data[cols_2_keep]

# Drop unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
#%%
#%%
if top_prot == True:
    # Select top 21 proteins
    top_ids = amp_sign['UniProt'].tolist()

    columns_to_drop = [
        col for col in data.columns
        if col.startswith('prot_') and not any(uniprot_id in col for uniprot_id in top_ids)
    ]

    # Drop columns
    data = data.drop(columns=columns_to_drop)


# Print teh number of columns
print(f"Data shape: {data.shape}")
#%%
# Code categorical variables appropriately
if frs == True:
    cat_vars = ['demo_sex_FRS', 'demo_smoking_FRS', 'demo_diabetes_FRS', 'demo_drugs_FRS']

if frs == False:
    cat_vars = ['demo_sex_FRS', 'demo_smoking_FRS', 'demo_diabetes_FRS', 'demo_alcohol', 'demo_drugs_FRS', 'demo_parental', 'demo_ethnicity']

if demo == True:  # Check if the passed argument 'demo' is True
    for var in cat_vars:
        # Apply transformation: 1 if value is 1, else 0
        data[var] = data[var].apply(lambda x: 1 if x == 1 else 0)

        # Code the categorical variables as categorical
        data[cat_vars] = data[cat_vars].astype('category')

# Select not categorical variables
not_cat_vars = data.columns[~data.columns.isin(cat_vars)]

# Drop ID, time and event columns from not_cat_vars
not_cat_vars = not_cat_vars[~not_cat_vars.isin(['ID', 'time', 'event'])]

#%%
if demo == True:  # Check if the passed argument 'demo' is True
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), not_cat_vars),
            ('cat', 'passthrough', cat_vars)
        ]
    )
    # Reconstruct the column names for transformed data
    all_columns = list(not_cat_vars) + list(cat_vars)  # Combine numerical and categorical column names
else:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), not_cat_vars)
        ]
    )
    # Reconstruct the column names for numerical data only
    all_columns = list(not_cat_vars)

#%%

# Define function to perform k-fold cross-validation
def custom_kfold(
        data_for_tune,
        lambda_to_test,  # lambda for lasso
        seed,
        folds
):

    # get the event column from y_train
    mace_col = data_for_tune.loc[:, 'event']

    # print the lambda to test
    print(f"Lambda to test: {lambda_to_test}")

    # Split the data into k-folds
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    # Initialize lists to store results
    c_index_list_train = []
    c_index_list_test = []

    # Loop over each fold
    for train_index, test_index in kf.split(data_for_tune, mace_col):
        # Split the data
        data_train, data_test = data_for_tune.iloc[train_index], data_for_tune.iloc[test_index]

        # Split the data into features and target
        x_train = data_train.drop(['time', 'event'], axis=1)
        y_train = data_train[['event', 'time']]
        x_test = data_test.drop(['time', 'event'], axis=1)
        y_test = data_test[['event', 'time']]

        # Save index of x_train and x_test
        x_test_index = x_test.index
        x_train_index = x_train.index

        prot_columns = x_train.columns

        # Scale the data (omit categorical variables)
        x_train = preprocessor.fit_transform(x_train)
        x_test = preprocessor.transform(x_test)

        x_train = pd.DataFrame(x_train, columns=prot_columns)
        if demo == True:
            x_test = pd.DataFrame(x_test, columns=list(not_cat_vars) + list(cat_vars))
        else:
            x_test = pd.DataFrame(x_test, columns=list(not_cat_vars))

        # Make x_train and x_test dataframes
        x_train = pd.DataFrame(x_train, columns=prot_columns)
        x_test = pd.DataFrame(x_test, columns=prot_columns)

        # Change index of x_train and x_test
        x_train.index = x_train_index
        x_test.index = x_test_index

        # Add y_train back to x_train
        #data_train_2 = pd.concat([x_train, y_train], axis=1)
        #data_test_2 = pd.concat([x_test, y_test], axis=1)

        # Fit the model
        model = CoxPHFitter(l1_ratio=1.0, alphas=[lambda_to_test])
        y_train_structured = np.array(list(y_train.itertuples(index=False)),
                                      dtype=[('event', 'bool'), ('time', 'float')])
        model.fit(x_train, y_train_structured)

        # Calculate concordance index
        c_index_train = concordance_index_censored(
            y_train_structured['event'],
            y_train_structured['time'],
            model.predict(x_train))[0]

        y_test_structured = np.array(list(y_test.itertuples(index=False)),
                                      dtype=[('event', 'bool'), ('time', 'float')])

        c_index_test = concordance_index_censored(
            y_test_structured['event'],
            y_test_structured['time'],
            model.predict(x_test))[0]

        # Append the results to the lists
        c_index_list_train.append(c_index_train)
        c_index_list_test.append(c_index_test)

    # Average the results from all folds
    c_index_mean_test = np.mean(c_index_list_test)
    c_index_mean_train = np.mean(c_index_list_train)

    # return data frame
    return pd.DataFrame({'lambda': lambda_to_test,
                         'c_index_train': c_index_mean_train,
                         'c_index_test': c_index_mean_test}, index=[0])
#%%

# define  l1
l1 = 10 ** np.linspace(-5, -1, num=50)

# Include 0 explicitly
l1 = np.concatenate(([0], l1))
#%%
# Create a data frame to store validation metrics
validation_metrics = pd.DataFrame(columns=['lambda',
                                           'c_index_train',
                                           'c_index_test'])

# Create a data frame to store coefficients
coefficients = pd.DataFrame(columns=['lambda', 'coefficient'])

# Create a data frame to store predictions
predictions = pd.DataFrame(columns=['ID', 'time', 'event', 'prediction', 'partial_hazard'])

# Create a df to store beta coefficients
prot_columns = data.columns[~data.columns.isin(['ID', 'time', 'event'])]

coef_df = pd.DataFrame(index=prot_columns, columns=['beta'])
#%%
# Get X and y separately
X = data.drop(['time', 'event'], axis=1)
y = data[['event', 'time']]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = (
    train_test_split(X, y,
                     test_size=0.2,
                     random_state=6220*random_seed,
                     shuffle=True,
                     stratify=y['event']))

# Export test and train studyid to txt using the path_to_results, the random_seed and the split number
X_test_studyid = X_test.loc[:, "ID"]
X_train_studyid = X_train.loc[:, "ID"]

X_test_studyid.to_csv(f'{path_to_results}/test_studyid_{random_seed}_{no_folds}folds.txt')
X_train_studyid.to_csv(f'{path_to_results}/train_studyid_{random_seed}_{no_folds}folds.txt')

# Drop ID column
X_test = X_test.drop("ID", axis=1)
X_train = X_train.drop("ID", axis=1)

# Scale the data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Export the tranformer
joblib.dump(preprocessor, f'{path_to_results}/preprocessor_{random_seed}_{no_folds}folds.pkl')
#%%
# Transform back to dataframes
# Transform back to dataframes
X_train = pd.DataFrame(X_train, columns=all_columns, index=X_train_studyid)
X_test = pd.DataFrame(X_test, columns=all_columns, index=X_test_studyid)

X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

# Merge X_train and y_train
data_tune = pd.concat([X_train, y_train], axis=1)

X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# Merge X_test and y_test
data_val = pd.concat([X_test, y_test], axis=1)

# Add ID column to data_val and data_tune by resetting the index
data_val.reset_index(drop=True, inplace=True)
data_tune.reset_index(drop=True, inplace=True)
#%%
no_iterations = len(l1)

# Run grid search
reff = Parallel(n_jobs=29)(delayed(custom_kfold)(data_for_tune=data_tune,
                                                 lambda_to_test=l1_i,
                                                 seed=random_seed*6220,
                                                 folds=no_folds) for l1_i in tqdm(l1, total=no_iterations))
grid_res = pd.concat(reff)

# Save the results
grid_res.to_csv(f'{path_to_results}/grid_res_lasso_{random_seed}_{no_folds}folds.csv')

# Save the lambda with the highest c-index on the test set
best_params = grid_res.query('c_index_test == c_index_test.max()').to_numpy()
best_lambda = best_params[0][0]

# Fit the model
# noinspection PyTypeChecker
best_model = CoxPHFitter(l1_ratio=1.0,
                         alphas=[best_lambda],
                         fit_baseline_model=True)
y_test_structured = np.array(list(y_test.itertuples(index=False)), dtype=[('event', 'bool'),('time', 'float')])
y_train_structured = np.array(list(y_train.itertuples(index=False)), dtype=[('event', 'bool'),('time', 'float')])
best_model.fit(X_train, y_train_structured)
#%%
# Save the model
joblib.dump(best_model, f'{path_to_results}/trained_model_{random_seed}_{no_folds}folds.pkl')

# Extract the coefficients
coefficients = best_model.coef_

# Save the coefficients
coef_df.iloc[:, 0] = coefficients[:, 0]

c_index_val = concordance_index_censored(y_test_structured["event"],
                                         y_test_structured["time"],
                                         best_model.predict(X_test))[0]

# Bootstrap the c-index for the validation set 1000 times
c_index_val_boot = []
for i in range(1000):
    data_val_boot = resample(data_val, replace=True, random_state=i*random_seed)

    # Separate X and y
    X_boot = data_val_boot.drop(columns=["event", "time"])  # Predictors
    y_boot = np.array(
        list(data_val_boot[["event", "time"]].itertuples(index=False)),
        dtype=[("event", "bool"), ("time", "float")]
    )

    c_index_val_boot.append(concordance_index_censored(y_boot["event"], y_boot["time"],
                                                       best_model.predict(data_val_boot.drop(['time', 'event'],
                                                                                             axis=1)))[0])

# Save the c-index for the validation set
c_index_val_boot = pd.DataFrame(c_index_val_boot, columns=['c_index'])
c_index_val_boot.to_csv(f'{path_to_results}/c_index_val_boot_lasso_{random_seed}_{no_folds}folds.csv')

# Save the c-index on the validation metrics data frame
validation_metrics = pd.concat([validation_metrics, pd.DataFrame({'lambda': best_lambda,
                                                                  'c_index_train': best_params[0][1],
                                                                  'c_index_test': best_params[0][2],
                                                                  'c_index_val': c_index_val},
                                                                 index=[0])])

# Calculate the 10-year survival probability and then the risk of dying
survival_at_10_years = best_model.predict_survival_function(X_test)
survival_at_10_years_values = np.array([sf(10*365) for sf in survival_at_10_years])
risk_of_dying_at_10_years = 1 - survival_at_10_years_values

# Calculate partial hazard
partial_hazard = best_model.predict(X_test)

# Calculate log partial hazard
log_partial_hazard = np.exp(partial_hazard)

# Save the predictions
predictions = pd.concat([
    predictions,
    pd.DataFrame({
        'ID': X_test_studyid.reset_index(drop=True),  # Reset index here
        'time': data_val['time'].reset_index(drop=True),  # Reset index here
        'event': data_val['event'].reset_index(drop=True),  # Reset index here
        'prediction': risk_of_dying_at_10_years,
        'partial_hazard': partial_hazard,
        'log_partial_hazard': log_partial_hazard
    })
]).reset_index(drop=True)  # Final reset after concatenation

# Add test studyid to predictions but first reset the index
X_test_studyid.reset_index(drop=True, inplace=True)
predictions['ID'] = X_test_studyid

# Save the validation metrics
validation_metrics.to_csv(f'{path_to_results}/validation_metrics_lasso_{random_seed}_{no_folds}folds.csv')

# Save the predictions
predictions.to_csv(f'{path_to_results}/predictions_lasso_{random_seed}_{no_folds}folds.csv')

# Save the coefficients
coef_df.to_csv(f'{path_to_results}/coefficients_lasso_{random_seed}_{no_folds}folds.csv')
#%%
# Define FPR thresholds (5-40%)
fpr_thresholds = np.linspace(0.05, 0.4, 8)

res_dr_lr_fpr = []

for fpr_threshold in fpr_thresholds:
    # Define threshold for risk prediction
    risk_threshold = np.quantile(predictions['partial_hazard'], (1 - fpr_threshold))

    # Find high-risk patients
    high_risk_patients = predictions['ID'][predictions['partial_hazard'] >= risk_threshold]

    # Calculate true positive rate (TPR)
    true_positives = predictions['ID'][(predictions['partial_hazard'] >= risk_threshold) & (predictions['event'] == 1)]
    false_positives = predictions['ID'][(predictions['partial_hazard'] >= risk_threshold) & (predictions['event'] == 0)]
    total_events = predictions['event'].sum()
    total_non_events = predictions['event'].count() - total_events

    # Calculate true positive rate (TPR)
    tpr = len(true_positives) / total_events

    # Calculate false positive rate (FPR)
    fpr = len(false_positives) / total_non_events

    # Calculate likelihood ratio
    lr = tpr / fpr

    # Append results to list
    res_dr_lr_fpr.append({
        'fpr_threshold': fpr_threshold,
        'risk_threshold': risk_threshold,
        'dr': tpr,
        'fpr': fpr,
        'lr': lr
    })

# Convert to DataFrame
res_dr_lr_fpr = pd.DataFrame(res_dr_lr_fpr)

# Save the results
res_dr_lr_fpr.to_csv(f'{path_to_results}/res_dr_lr_fpr_lasso_{random_seed}_{no_folds}folds.csv')
