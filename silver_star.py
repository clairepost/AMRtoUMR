##Plan:
#1) Load in all of the data (start with a subset)
#2) Split the train and tests into the necessary folds
#3) Augment folds with the training data:
#4) Run NN models 1 and 2
#5) predict and store results under silver-star-k_folds/fold



#silver_star all data will look like this for each fold and then one where all fodls are combined
#sent, amr_role, umr_role, y_pred_RULE, y_pred_NN, y_pred_NN_RULE

#This will allow for the easiest comparison

from helper_functions import preprocess_data, get_embeddings
import pandas as pd
import torch
from error_analysis import get_indices
from baseline import basic_baseline
from base_nn import basic_base_nn
from nn_with_rules_weights import basic_nn_with_rules_w

file_path = "output/silver-star-k_folds/"

def run(num_epochs):

    output_file_path = file_path + str(num_epochs) + "/"
   
    silver_data = preprocess_data("silver", False, False)
    augment_data= preprocess_data("augment2", False, False)
    gold1 = preprocess_data("train", False, False)
    gold2 = preprocess_data("test", False, False)

    gold_data = pd.concat([gold1, gold2, augment_data], axis = 0)
    gold_data = gold_data.dropna(axis = 1)
    gold_data.to_csv(output_file_path + "gold_data.csv")
    silver_data.to_csv(output_file_path + "silver_data.csv")



    #REMOVE THIS LINE FOR RUNNING THE WHOLE THING
    silver_data = silver_data.head(10)
    silv_embeddings = get_embeddings(silver_data)
    gold_embeddings = get_embeddings(gold_data)

    splits= get_indices(gold_data["umr_role"])

    full_df = gold_data.head(0)
    for i, (train_index, test_index) in splits:
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")

        #Create fold -specific data
        fold_train = gold_data.iloc[train_index]
        fold_train = pd.concat([fold_train, silver_data], axis = 0)
        fold_test= gold_data.iloc[test_index]

        fold_embeddings_train = gold_embeddings[train_index]
        fold_embeddings_train = torch.cat((fold_embeddings_train, silv_embeddings), dim=0)
        fold_embeddings_test = gold_embeddings[test_index]

        #RULES
        print("UPDATING RULES")
        fold_test["y_pred_RULE"] =  basic_baseline(fold_test)

        #intermediate save
        
        fold_test.to_csv(f'{output_file_path}fold_{i}.csv')

        #BASIC NN
        print(fold_train)
        print("RUNNING BASE NN")
        
        fold_test["y_pred_NN"] = basic_base_nn(fold_train, fold_test, fold_embeddings_train, fold_embeddings_test, num_epochs)
        #intermediate save
        fold_test.to_csv(f'{output_file_path}fold_{i}.csv')

        #NN WITH RULES
        print("RUNNING NN + RULES")
        fold_test["y_pred_NN_RULE"] = basic_nn_with_rules_w(fold_train, fold_test, fold_embeddings_train, fold_embeddings_test, num_epochs)

        #intermediate save
        fold_test.to_csv(f'{output_file_path}fold_{i}.csv')

        #SAVE all into one DF
        fold_test["fold"] = i
        full_df = pd.concat([full_df, fold_test], axis = 0)


    #save full df
    full_df.to_csv(f'{output_file_path}all_folds.csv')
    print("SILVER FINISHED")
    

def run_stats():
    pass
    #open up the csv for all folds. split by fold
    #compute macro F1, micro F1
    #report dataset statistics training data has all these different roles and these counts
    #show k-fold stuff
    #show per-class performance (aggregate?)



if __name__ == "__main__":
    epochs_list = [10]
    for epoch in epochs_list:
        run(epoch)
    run_stats()
