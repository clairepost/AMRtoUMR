
from helper_functions import extract_data
from rules import detect_split_role
import os
import random
import pandas as pd
##The baeline model
# this will be a rules-only baseline model, extract the test data (animacy info will already be applied)
def calc_role(Y_preds):
    #given a rule prediction with weights, return a list consisting of the most probable of the rules given
    Y_prob = []
    for i in range(len(Y_preds)):
        labels, values = Y_preds[i]
        result_label = random.choices(labels, weights=values, k=1)[0]
        Y_prob.append(result_label)
    return Y_prob



def run_baseline():
    #extract test data into X,Y
    #run animacy_parser on X, to get tuples
    x_file = "x_test.csv"
    y_file = "y_trues_test.csv"
    out_file = "baseline_train.csv"
    # if os.path.exists(x_file) and os.path.exists(y_file):
    #     X= pd.read_csv(x_file)
    #     #X_tuples = list(X.to_records(index=False))
    #     y_true= pd.read_csv(y_file)  
        
   # else: 
    X_tupes, y_true = extract_data(True) #setting this argument equal to False means we are getting test data
    X =  pd.DataFrame.from_records(X_tupes, columns = ['sent','ne_info' ,'amr_graph','amr_head_name', 'amr_role', 'amr_tail_name'])
        #X.to_csv(x_file)
        #y_true.to_csv(y_file)
    #apply rules to X to get predictions
    y_pred = detect_split_role(X) #gets roles and weights
    y_pred = calc_role(y_pred) # just picks most prob role


    #output preditcions
    y_pred= pd.Series(y_pred, name='y_rules')
    df = pd.concat([y_true, y_pred], axis=1)
    df.to_csv(out_file)


    
    
    #return predictions
    return y_pred,y_true


def compare_results(y_true,y_pred):
    pass

y_true, y_pred = run_baseline()
compare_results(y_true,y_pred)
