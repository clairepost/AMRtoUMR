import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report,f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
#using stratified k-fold cross validation

# def run_buckets(train, test):
#     # combine training and test data
#     # using a seed- split into 5 different 80-20 splits?
#     # for each bucket: train model and test
#         # report accuracy
#         #report the final layer of the weights - that's how much we should be able to trust the model
#         # report confusion matrix
#     pass



def get_indices(x):
    num_splits = 5
    seed = 0
    kf = StratifiedKFold(n_splits=num_splits, shuffle = True, random_state=seed)
    kf.get_n_splits(x,x)
    print(x)
    return enumerate(kf.split(x,x))


def create_line_graph(results_df, output_file_name):
    # Line Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Fold', y='F1', hue='Model', data=results_df, marker='o')
    plt.title('Model Performance Across Folds')
    plt.xlabel('Fold')
    plt.xticks(range(0,5))
    plt.ylabel('Macro F1')
    plt.show()
    plt.savefig("results/" + output_file_name)


def get_macro_f1(df):
    # Extract values from DataFrame columns
    y_true = df['umr_role'].values
    y_preds_list = df['y_pred'].values

    # Calculate accuracy

    if type(y_preds_list[0]) == list:
        y_preds_transposed = list(map(list, zip(*y_preds_list)))
        
        # Compute accuracy for each set of predictions
        f1_scores = [f1_score(y_true, y_preds,average="macro") for y_preds in y_preds_transposed]
        

        # Compute average accuracy
        f1 = np.mean(f1_scores) 
    else:
        f1 = f1_score(y_true, y_preds_list,average="macro")
        #print(confusion_matrix(y_preds_list,y_true))
    return f1

def create_results_df(directory_path):
    #creates the results_df -where  each filepath has a model_name and fold
    #directory_path = "output/k-foldv2/"
    # List to store results
    results = []


    file_name = 'all_folds.csv'
    # Construct the full file path
    all_folds_file_path = os.path.join(directory_path, file_name)

    # Chek if the the all_folds_file exists
    if os.path.exists(all_folds_file_path):
        df = pd.read_csv(all_folds_file_path)
        # Group the DataFrame by the 'fold' column
        grouped = df.groupby('fold')

        # Initialize an empty list to store results
        results = []

        # Iterate over each group
        for fold, group in grouped:
            umr_role = group['umr_role']
            for model_name in ['RULE', 'NN', 'NN_RULE']:  # Replace with your model names
                y_pred = group[f'y_pred_{model_name}']
                f1 = f1_score(umr_role, y_pred,average="macro")
                results.append({'Model': model_name, 'Fold': fold, 'F1': f1})

    else:
        # Iterate through files in the directory
        for file_name in os.listdir(directory_path):
            if file_name.endswith(".csv"):
                # Extract model and fold information from the file name
                model_fold = file_name.rsplit("_", 1)
                model = model_fold[0]
                fold = model_fold[1].split(".")[0]  # Remove the '.csv' extension
                fold = int(fold)

                # Read the CSV file into a DataFrame
                file_path = os.path.join(directory_path, file_name)
                df = pd.read_csv(file_path)

                # Calculate accuracy using your custom function
                f1 = get_macro_f1(df)

                # Append the results to the list
                results.append({'Model': model, 'Fold': fold, 'F1': f1})

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results)
    average_f1_scores = results_df.groupby('Model')['F1'].mean().reset_index()
    return results_df

def compute_macro_folds(results_df):
    average_f1_scores = results_df.groupby('Model')['F1'].mean().reset_index()
    return average_f1_scores


def create_classificaiton_report(directory_path, results_path):
    file_name = 'all_folds.csv'
    # Construct the full file path
    all_folds_file_path = os.path.join(directory_path, file_name)

    classification_reports = {}

    # Chek if the the all_folds_file exists
    if os.path.exists(all_folds_file_path):
        df = pd.read_csv(all_folds_file_path)
        for model_name in ['RULE', 'NN', 'NN_RULE']:  # Replace with your model names
            y_pred = df[f'y_pred_{model_name}'].dropna(axis=0)

            classification_reports[model_name] = classification_report(df['umr_role'].dropna(axis=0), y_pred, output_dict=True)


    # Combine the reports into a single DataFrame
    combined_df = pd.DataFrame()
    support = {}

    # Iterate over the classification reports dictionary
    for model, report in classification_reports.items():
        print(report)
        # Extract the F1 scores for each label and create a DataFrame
        f1_scores = {}
        for label , metrics in report.items():
            if type(report[label]) == dict:
                f1_scores[label] = report[label]["f1-score"]
                support[label] = report[label]["support"]
            else:
                f1_scores[label] = report[label]
        model_df = pd.DataFrame.from_dict({model: f1_scores}, orient='index')
        # Concatenate the DataFrame with the combined DataFrame
        combined_df = pd.concat([combined_df, model_df])

    # Reset index to make model names a column
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'index': 'Model'}, inplace=True)

    # Melt the DataFrame to have labels as rows
    combined_df = pd.melt(combined_df, id_vars=['Model'], var_name='Label', value_name='F1')

    # Pivot the DataFrame
    reshaped_df = combined_df.pivot(index='Label', columns='Model', values='F1').reset_index()

    df2 = pd.DataFrame.from_dict(support, orient='index').reset_index()
    df2.columns = ['Label', 'support']  # assuming the columns match your original DataFrame

    # Concatenate the DataFrames along the column axis
    combined_df = pd.concat([reshaped_df.set_index('Label'), df2.set_index('Label')], axis=1).reset_index()

    # Display the reshaped DataFrame
    print(combined_df)

    # Display the combined DataFrame
    combined_df.to_csv("results/" + results_path)

def create_all_folds(folder_path):

    # Initialize an empty list to store DataFrames
    dfs = []
    # Initialize a set to store unique model names
    model_names = set()

    # Iterate over the files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv') and file_name != "all_folds.csv":
            # Read the CSV file into a DataFrame
            df = pd.read_csv(os.path.join(folder_path, file_name))
            # Extract the model name and fold number from the file name
            model_name, fold = file_name.split('_')[0], file_name.split('_')[1].split('.')[0]
            # Add the model name to the set of unique model names
            model_names.add(model_name)
            # Add a 'model_name' column and a 'fold' column
            df["y_pred" + model_name] = df["y_pred"]
            df['fold'] = fold
            df.set_index("Unnamed: 0")
            #df = df.drop("y_pred")
            # Append the DataFrame to the list
            dfs.append(df)

    # Concatenate all DataFrames into one
    for i in range(len(dfs)):
        print((dfs[i].columns))
    combined_df = dfs[0].join(dfs[1:])


    # Display the combined DataFrame
    print(combined_df)
    combined_df.to_csv(folder_path + "/all_folds.csv")



if __name__ == "__main__":
    # file_path = "output/silver-star-k_folds/50"
    # k_fold_silver = create_results_df(file_path)
    # print(k_fold_silver)
    # print(compute_macro_folds(k_fold_silver))
    # create_classificaiton_report(file_path, "k-fold_silver-classifcation.csv")
    # create_line_graph(k_fold_silver,"k-fold_silver.png")
    #create_all_folds("output/k-foldv2")
    create_classificaiton_report("output/k-foldv2", "k-fold_gold-classifcation.csv")
   
