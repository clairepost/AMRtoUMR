
from helper_functions import extract_data, preprocess_data
from rules import detect_split_role
import os
import random
import pandas as pd
import ast

##The baseline model
# this will be a rules-only baseline model, extract the test data (animacy info will already be applied)
def calc_role(Y_preds):
    print("calc roles")
    #given a rule prediction with weights, return a list consisting of the most probable of the rules given
    Y_prob = []
    for i in range(len(Y_preds)):
        labels, values = Y_preds[i]
        result_label = random.choices(labels, weights=values, k=1)[0]
        Y_prob.append(result_label)
    return Y_prob


def run_baseline_X_times():
    print("running baseline x times")
    num_iterations=20
    # X_tupes, y_true = extract_data("test")
    print("Running augment baseline")
    X_tupes, y_true = extract_data("augment")
    X = pd.DataFrame.from_records(X_tupes, columns=['sent', 'ne_info', 'amr_graph', 'amr_head_name', 'amr_role', 'amr_tail_name'])
    
    comparison_results = []
    y_pred = detect_split_role(X)
    for _ in range(num_iterations):
        y_prob = calc_role(y_pred)
        
        comparison_result = compare_results(y_true, y_prob)
        comparison_results.append(comparison_result)

    # Save comparison results to the DataFrame
    for i, result in enumerate(comparison_results):
        df[f'comparison_results_{i}'] = result

    # Save the DataFrame to a new CSV file
    df.to_csv(f'baseline_results_debug.csv', index=False)
    return comparison_results


def compare_results(y_true, y_pred):
    print("compare results")
    comparison_results = []
    correct_count = 0

    for true_label, pred_label in zip(y_true, y_pred):
        # Skip comparison for empty values in y_true
        if not pd.isnull(true_label):
            result = 1 if true_label == pred_label else 0
            comparison_results.append(result)
            correct_count += result

    if len(y_true) - y_true.isnull().sum() > 0:
        accuracy = correct_count / (len(y_true) - y_true.isnull().sum())
        print(f'Accuracy: {accuracy}')
    else:
        print("No non-empty values in y_true.")

def run_baseline(num_iters, split):
    print("running baseline")
    
    #slightly different implementation than what is happening in compare_results. Just so the format is consistent across models
    #takes in num of iterations: returns X (input data- size (n x m)) and y_probs (size n x num_iters)
    y_preds = []
    X = preprocess_data(split, False, False) #reload the graphs and the rules -> set both to True
    y_prob = list(zip(X["y_guess"], X["y_guess_dist"]))
    for i in range(num_iters):
        y_calc = calc_role(y_prob)
        y_preds.append(y_calc)

    c = len(y_preds)
    n = len(y_preds[0])

    #print(y_preds)
    # Transpose the list
    list_of_lists_n = [[y_preds[j][i] for j in range(c)] for i in range(n)]

    y_pred = pd.Series(list_of_lists_n,name ="y_pred") 
    full_df = pd.concat([X, y_pred],axis = 1)
    full_df.to_csv("output/470-results/baseline_"+split+"_files_470.csv")
    print("finished running baseline over", num_iters, "splits")
    return full_df

def run_splits_nn():
    num_iters = 1
    X_1 = preprocess_data("train",False,False)
    X = preprocess_data("test", False, False)
    X_2 = preprocess_data("augment2", False, False)

    all_Xs = pd.concat((X_1,X, X_2),axis=0)
    all_Xs = all_Xs.dropna(axis =1)

    splits= get_indices(all_Xs["umr_role"])
    print(splits)

    for i, (train_index, test_index) in splits:
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        
        y_preds = []
        X = all_Xs.iloc[test_index.tolist()]
        y_prob = list(zip(X["y_guess"], X["y_guess_dist"]))
        for _ in range(num_iters):
            print(y_prob)
            y_calc = calc_role(y_prob)
            y_preds.append(y_calc)

        c = len(y_preds)
        n = len(y_preds[0])

        print(y_preds)
        # Transpose the list
        list_of_lists_n = [[y_preds[j][i] for j in range(c)] for i in range(n)]
       # y_pred = pd.Series(list_of_lists_n,name ="y_pred") 
    
        #X = X.loc[:,~X.columns.duplicated()].copy() #drop duplicte columns
        print(type(y_preds[0]))
        print(y_preds[0])
        #y_preds = pd.Series(y_preds)
        X["y_pred"] = y_preds[0]
        X.to_csv("output/k-foldv2/baseline_"+str(i)+".csv")
        print("finished running k-fold baseline rules")
    return X

def combine_baseline_results():
    train_df = run_baseline(5, "train")
    test_df = run_baseline(5,"test")
    full_df = pd.concat([train_df,test_df],axis = 0)
    full_df.to_csv("output/470-results-inverted/baseline.csv")


def basic_baseline(X):
    y_prob = list(zip(X["y_guess"], X["y_guess_dist"]))
    y_calc = calc_role(y_prob)

    return y_calc

if __name__ == "__main__":
    df = pd.DataFrame()
    # Run the baseline model
    #run_baseline_X_times()
   # run_splits_nn()
 #   run_baseline(5,"augment2") #Marie's version
    combine_baseline_results()
    
