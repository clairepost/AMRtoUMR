##The baeline model
# this will be a rules-only baseline model, extract the test data (animacy info will already be applied)
from helper_functions import extract_data


def run_baseline():
    #extract test data into X,Y
    #run animacy_parser on X, to get tuples
    X_tupes, y_true = extract_data(False) #setting this argument equal to False means we are getting test data
    y_pred = ""
    y_true = ""
    #apply rules to X to get predictions

    #return predictions
    return y_pred,y_true


def compare_results(y_true,y_pred):
    pass

y_true, y_pred = run_baseline()
compare_results(y_true,y_pred)
