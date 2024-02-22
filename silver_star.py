##Plan:
#1) Load in all of the data (start with a subset)
#2) Split the train and tests into the necessary folds
#3) Augment folds with the training data:
#4) Run NN models 1 and 2
#5) predict and store results under silver-star-k_folds/fold



#silver_star all data will look like this for each fold and then one where all fodls are combined
#sent, amr_role, umr_role, y_pred_RULE, y_pred_NN, y_pred_NN_RULE

#This will allow for the easiest comparison

from helper_functions import preprocess_data
import pandas as pd
from error_analysis import get_indices
from baseline import basic_baseline
from base_nn import basic_base_nn

def run():
    output_file_path = "output/silver-star-k_folds/"

    silver_data = preprocess_data("silver", False, False)
    augment_data= preprocess_data("augment2", False, False)
    gold1 = preprocess_data("train", False, False)
    gold2 = preprocess_data("test", False, False)

    gold_data = pd.concat([gold1, gold2, augment_data], axis = 0)
    gold_data.to_csv(output_file_path + "gold_data.csv")

    #REMOVE THIS LINE FOR RUNNING THE WHOLE THING
    silver_data = silver_data.head(10)

    splits= get_indices(gold_data["umr_role"])

    full_df = gold_data.head(0)
    for i, (train_index, test_index) in splits:

        #define train data
        fold_train = gold_data.iloc[train_index]
        fold_train = pd.concat([fold_train, silver_data], axis = 0) 
        #define test data
        fold_test= gold_data.iloc[test_index]

        #RULES
        fold_test["y_pred_RULE"] =  basic_baseline(fold_test)

        #intermediate save
        fold_test.to_csv(f'{output_file_path}fold_{i}.csv')

        #BASIC NN
        print(fold_train)
        fold_test["y_pred_NN"] = basic_base_nn(fold_train, fold_test)
        #intermediate save
        fold_test.to_csv(f'{output_file_path}fold_{i}.csv')

        #NN WITH RULES
        fold_test["y_pred_NN_RULE"] = 3


        #intermediate save
        fold_test.to_csv(f'{output_file_path}fold_{i}.csv')

        #SAVE all into one DF
        fold_test["fold"] = i
        full_df = pd.concat([full_df, fold_test], axis = 0)


    #save full df
    full_df.to_csv(f'{output_file_path}all_folds.csv')
    print("SILVER FINISHED")
    





if __name__ == "__main__":
    run()
