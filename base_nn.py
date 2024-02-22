#This function will create a base NN 
#Steps
#1 : preprocess input data to just be the sentence embedding and one hot encoding of the relation
#2 : create the neural network. The neural network architecture allows for contraining the output depending on the feature role that we provide

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from helper_functions import get_embeddings, create_mapping, preprocess_data
from error_analysis import get_indices
import numpy as np
import ast

def preprocessing_for_NN(split, reload_data = True, X = [], embeddings = []):
    #split is either "train" or "test"
    #load in data, get bert embeddings, and set it up as tensors

    # if split == "augmented":
    #     X = pd.read_csv("input_data/annotated_470.csv")
    #     X['ne_info'] = X['ne_info'].apply(ast.literal_eval) #ne_info will need to be a literal
    #     X['umr_role'] = X['y_gold']

    # else:
    if reload_data == True:
        X = preprocess_data(split, False, False)
    
    mapping ,swap_amr_int_dict,swap_umr_int_dict = create_mapping()

    # Convert the categorical column to numerical form using the mapping
    column_type = X['amr_role'].dtype
    column_type = X['umr_role'].dtype


    X['amr_role'] = X['amr_role'].map(swap_amr_int_dict)
    print(X['umr_role'])
    X['umr_role'] = X['umr_role'].map(swap_umr_int_dict)


    X = X.dropna(subset=["umr_role"])
    X = X.dropna(subset=["amr_role"])

    X.reset_index(drop=True)
    
    umr_role = torch.tensor(X["umr_role"].to_list(),dtype=torch.long)
    amr_role = torch.tensor(X['amr_role'].to_list(), dtype=torch.long)
    if embeddings == []:
        embeddings = get_embeddings(X)
    else:
        print("length of embeddings better equal length of data:", len(embeddings), len(X))
    

    # #print sizes of returned data
    # print("amr_role" , amr_role.size())
    # print("umr_role" , umr_role.size())
    # print("embeddings size", embeddings.size()) #size ([50,768])

    return embeddings,amr_role, umr_role, X, mapping, swap_umr_int_dict,swap_amr_int_dict #return X and y_truefor mapping back to the categories later

def train_model(embeddings, amr_role, umr_role,mapping, num_eps):

    # Sample data (replace this with your actual data)
    # X = torch.randn((100, 5))  # 100 samples, 5 features
    # letter = torch.randint(0, 3, (100,))  # 0: "a", 1: "b", 2: "c"
    # output = torch.randint(0, 4, (100,))  # Sample output (replace with your actual target)

    # print("sample x: ", X.size())
    # print("sameple_letter", letter.size())
    # print("sample output", output.size())

    dataset = TensorDataset(embeddings, amr_role, umr_role)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define the neural network model
    class CustomModel(nn.Module):
        def __init__(self, input_size, num_classes_output, num_amr_roles,mapping):
            super(CustomModel, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc_letter = nn.ModuleList([nn.Linear(64, num_classes_output) for _ in range(num_amr_roles)])
            self.mapping = mapping

        def forward(self, x, letter):
            x = torch.relu(self.fc1(x))
            output_branch = self.fc_letter[letter]

            allowed_outputs = torch.tensor(self.mapping[letter.item()]) #only allow whatever the amr_role maps to to becomne the output, letter is a tensor so we get the item

            output = output_branch(x)
            # Create a mask for indices not in the specified mask
            not_in_mask = ~torch.isin(torch.arange(len(output)), allowed_outputs)

            # Apply the desired operations using the mask
            tensor_result = torch.zeros_like(output)  # Initialize with zeros
            tensor_result[allowed_outputs] = output[allowed_outputs] * 1    # Multiply indices in the mask by 1
            tensor_result[not_in_mask] = float('-inf')  # Set indices not in the mask to -inf
            #output[:,~torch.tensor(allowed_outputs, dtype=torch.bool)] = float('-inf')  # Set disallowed outputs to -inf
            return tensor_result

    # Initialize the model
    input_size = embeddings.size(dim =1)  # Replace with the actual number of features
    num_amr_roles = len(mapping.keys())
    flat_values = [item for value in mapping.values() for item in (value if isinstance(value, list) else [value])]
    #num_classes_output = len(set(flat_values))
    num_classes_output =17 #hard-coded b/c I can't figure this out rn

    print("input size", input_size)
    print("num amr roles", num_amr_roles)
    print("num classes_ouptut",num_classes_output)

    model = CustomModel(input_size, num_classes_output, num_amr_roles, mapping)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = num_eps

    for epoch in range(num_epochs):
        for inputs, letter, targets in dataset:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, letter)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    return model


def predict(model, test_data, swap_umr_int_dict, swap_amr_int_dict):
    embeddings,amr_role, umr_role, X = test_data
    dataset = TensorDataset(embeddings, amr_role, umr_role)
    with torch.no_grad():
        predictions = []
        model.eval()
        for embeddings, amr_role, umr_role in dataset:
        # predict and swap it back from an integer to the class
            predictions.append(swap_umr_int_dict[torch.argmax(model(embeddings, amr_role)).item()])
   
    #convert the numbers back to categorical data


    y_preds = pd.Series(predictions, name = "y_pred")


    X['amr_role'] = X['amr_role'].map(swap_amr_int_dict)
    X['umr_role'] = X['umr_role'].map(swap_umr_int_dict)

  
    print(y_preds)
    X["y_pred"] = predictions

    return X


# Once trained, you can use the model for predictions
# Replace `your_input_data` with your actual input data and letter


# Convert the predictions to class labels if needed
#predicted_labels = torch.argmax(predictions, dim=1)

# original_categories = df['CategoryColumn'].cat.categories
# mapped_values = other_numerical_tensor.numpy().tolist()
# original_values = pd.Series(mapped_values).map(dict(enumerate(original_categories)))


# The predicted_labels are the predicted classes for your output
def run_base_nn():
    embeddings,amr_role, umr_role, X, mapping, swap_umr_int_dict, swap_amr_int_dict= preprocessing_for_NN("train")
    model = train_model(embeddings,amr_role, umr_role,mapping)

    #get the test data
    embeddings,amr_role, umr_role, X, mapping, swap_umr_int_dict,swap_amr_int_dict = preprocessing_for_NN("test")
    df_test = predict(model, (embeddings, amr_role, umr_role,X), swap_umr_int_dict, swap_amr_int_dict) 

    df_test.to_csv("output/base_nn_test.csv")
    return df_test


def run_splits_nn(model_choice):
    model_list = ["base_nn", "nn_with_rules_weights", "baseline"]
    if model_choice not in model_list:
        print("pick a model that has been created")
        return
    

    embeddings, amr_role, umr_role, X, mapping, swap_umr_int_dict, swap_amr_int_dict = preprocessing_for_NN("train")
    
    embeddings_1,amr_role_1, umr_role_1, X_1, mapping_1, swap_umr_int_dict_1,swap_amr_int_dict_1 = preprocessing_for_NN("test")
    embeddings_2, amr_role_2, umr_role_2, X_2, _,_,_ = preprocessing_for_NN("augment2")
    all_embeddings=  torch.cat((embeddings,embeddings_1, embeddings_2), 0)
    all_amr_roles = torch.cat((amr_role,amr_role_1, amr_role_2),0)
    all_umr_roles = torch.cat((umr_role, umr_role_1, umr_role_2),0)
    all_Xs = pd.concat((X,X_1, X_2),axis=0)


    splits= get_indices(all_umr_roles)

    for i, (train_index, test_index) in splits:
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        
        #select training data
        embeddings =  torch.index_select(all_embeddings, 0, torch.LongTensor(train_index))
        amr_roles = torch.index_select(all_amr_roles, 0,torch.LongTensor(train_index) )
        umr_roles = torch.index_select(all_umr_roles, 0,torch.LongTensor(train_index) )

        model = train_model(embeddings,amr_roles, umr_roles,mapping)

        #select test data
        embeddings =  torch.index_select(all_embeddings, 0, torch.LongTensor(test_index))
        amr_roles = torch.index_select(all_amr_roles, 0,torch.LongTensor(test_index))
        umr_roles = torch.index_select(all_umr_roles, 0,torch.LongTensor(test_index))

        Xs = all_Xs.iloc[test_index.tolist()]

        df_test = predict(model, (embeddings, amr_roles, umr_roles,Xs), swap_umr_int_dict, swap_amr_int_dict) 

        print(len(Xs))
        print(len(df_test))
        df_test.to_csv(f"output/k-foldv2/{model_choice}_test_{i}.csv")
        print("finished running k-folds pn base_nn")
    return df_test

def run_on_all_data(inverse= False):
    # creates the training data from the train data and test data, results in around 100 examples

    #inverse == False:  the splits to have train/test as the train and then evaluates on the augmented data
    if inverse == False:
        embeddings, amr_role, umr_role, X, mapping, swap_umr_int_dict, swap_amr_int_dict = preprocessing_for_NN("train")
        
        embeddings_1,amr_role_1, umr_role_1, X_1, mapping_1, swap_umr_int_dict_1,swap_amr_int_dict_1 = preprocessing_for_NN("test")
        all_embeddings=  torch.cat((embeddings,embeddings_1), 0)
        all_amr_roles = torch.cat((amr_role,amr_role_1),0)
        all_umr_roles = torch.cat((umr_role, umr_role_1),0)
        all_Xs = pd.concat((X,X_1),axis=0)

        model = train_model(all_embeddings,all_amr_roles, all_umr_roles,mapping)

        embeddings_1,amr_role_1, umr_role_1, X_1, mapping_1, swap_umr_int_dict_1,swap_amr_int_dict_1 = preprocessing_for_NN("augment2")

        df_test = predict(model, (embeddings_1, amr_role_1, umr_role_1,X_1), swap_umr_int_dict, swap_amr_int_dict) 
        df_test.to_csv(f"output/470-results/base_nn.csv")
    else:
        embeddings, amr_role, umr_role, X, mapping, swap_umr_int_dict, swap_amr_int_dict = preprocessing_for_NN("train")
        embeddings_1,amr_role_1, umr_role_1, X_1, mapping_1, swap_umr_int_dict_1,swap_amr_int_dict_1 = preprocessing_for_NN("test")
        all_embeddings=  torch.cat((embeddings,embeddings_1), 0)
        all_amr_roles = torch.cat((amr_role,amr_role_1),0)
        all_umr_roles = torch.cat((umr_role, umr_role_1),0)
        all_Xs = pd.concat((X,X_1),axis=0)

        embeddings_2,amr_role_2, umr_role_2, X_2, mapping_2, swap_umr_int_dict_1,swap_amr_int_dict_1 = preprocessing_for_NN("augment2")
        model = train_model(embeddings_2,amr_role_2, umr_role_2,mapping_2)

        df_test = predict(model, (all_embeddings, all_amr_roles,all_umr_roles,all_Xs), swap_umr_int_dict_1, swap_amr_int_dict_1)
        df_test.to_csv(f"output/470-results-inverted/base_nn.csv")
    
    print("Finished baseline_nn on all data")

def basic_base_nn(train,test,train_embeddings, test_embeddings, num_epochs):
    embeddings, amr_role, umr_role, X, mapping, swap_umr_int_dict, swap_amr_int_dict = preprocessing_for_NN("train", False, train, train_embeddings)
    embeddings_1,amr_role_1, umr_role_1, X_1, mapping_1, swap_umr_int_dict_1,swap_amr_int_dict_1 = preprocessing_for_NN("test", False, test, test_embeddings)

    print(amr_role, umr_role)
    model = train_model(embeddings,amr_role, umr_role,mapping, num_epochs)
    df_test = predict(model, (embeddings_1, amr_role_1, umr_role_1,X_1), swap_umr_int_dict, swap_amr_int_dict) 
    return df_test["y_pred"].to_list()



if __name__ == "__main__":
    run_on_all_data(inverse = True)
  #  run_splits_nn("base_nn")
    
