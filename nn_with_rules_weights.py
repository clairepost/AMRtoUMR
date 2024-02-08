#This function will create a NN with rules
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

import ast
import sklearn

def create_rule_weight_tensor(y_guess, rule_weights,mapping):
    #takes guess and weights and will return a tensor of size(n, output_length) for the corresponding weights of each rule
    flat_values = [item for value in mapping.values() for item in (value if isinstance(value, list) else [value])]
    num_classes_output = len(set(flat_values))//2
    weight_tensor = torch.zeros(len(y_guess),num_classes_output)
    #weight_tensor = torch.full((len(y_guess),num_classes_output),0.01) #allow for the rules to be mistaken here as well

    for i in range(len(y_guess)):
        for j in range(len(y_guess[i])):
            y_guess[i][j] = mapping[y_guess[i][j]]

    # Fill the tensor with corresponding weights
    for i, (class_numbers, weights_list) in enumerate(zip(y_guess, rule_weights)):
        for j in range(len(class_numbers)):
            weight_tensor[i, class_numbers[j]] = float(weights_list[j])

    return weight_tensor


def preprocessing(split):
    #training is a boolean: set it to trtue, to preprocess the training data, and false to preprocess the test data
    #load in data, get bert embeddings, and set it up as tensors

    X = preprocess_data(split, False, False)
   
    mapping ,swap_amr_int_dict,swap_umr_int_dict = create_mapping()

    # Convert the categorical column to numerical form using the mapping
    X['amr_role'] = X['amr_role'].map(swap_amr_int_dict)
    X['umr_role'] = X['umr_role'].map(swap_umr_int_dict)

    umr_roles = torch.tensor(X["umr_role"],dtype=torch.long)
    amr_roles = torch.tensor(X['amr_role'], dtype=torch.long)
    embeddings = get_embeddings(X) 
   

    y_guess = X["y_guess"]
    rule_weights = X["y_guess_dist"]
    rule_outputs = create_rule_weight_tensor(y_guess, rule_weights,swap_umr_int_dict)

    #print sizes of returned data
    
    print("class_weights:",rule_outputs.size())
    print("umr_role" , umr_roles.size())
    print("amr_role" , amr_roles.size())
    print("embeddings size", embeddings.size()) #size ([50,768])

    return embeddings,amr_roles, umr_roles, X, mapping, swap_umr_int_dict,swap_amr_int_dict, rule_outputs #return X and y_truefor mapping back to the categories later

def train_model(embeddings, amr_roles, umr_roles,mapping, rule_outputs):

    # #define the loss function
    # class CustomLoss(nn.Module):
    #     def __init__(self):
    #         super(CustomLoss, self).__init__()
    #     def forward(self,y_pred,y_true, class_weight):
    #         loss = nn.CrossEntropyLoss()
    #         supervised_loss = loss(y_pred, y_true)
    #         weighted_supervised_loss = supervised_loss * class_weight
    #         total_loss = torch.mean(weighted_supervised_loss)
    #         return total_loss

    # Define the neural network model
    class CustomModel(nn.Module):
        def __init__(self, input_size, num_classes_output, num_amr_roles,mapping):
            super(CustomModel, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc_amr_role = nn.ModuleList([nn.Linear(64, num_classes_output) for _ in range(num_amr_roles)])
            self.mapping = mapping
            self.weights = nn.Linear(2,2)  # Updated to match the number of classes
            # Initialize the weights to 0.5 for all entries
            nn.init.constant_(self.weights.weight, 0.5)
        def forward(self, x, amr_role, rule_output):
            #input is x- input data, letter -amr_role for split into diff module,rule_output - rules output 
            x = torch.relu(self.fc1(x))
            output_branch = self.fc_amr_role[amr_role]

            allowed_outputs = torch.tensor(self.mapping[amr_role.item()]) #only allow whatever the amr_role maps to to becomne the output, letter is a tensor so we get the item

            output = output_branch(x)
            # Create a mask for indices not in the specified mask
            not_in_mask = ~torch.isin(torch.arange(len(output)), allowed_outputs)

            # Apply the desired operations using the mask
            tensor_result = torch.zeros_like(output)  # Initialize with zeros
            tensor_result[allowed_outputs] = output[allowed_outputs] * 1    # Multiply indices in the mask by 1
            tensor_result[not_in_mask] = float(0)  # Set indices not in the mask to -inf
            #output[:,~torch.tensor(allowed_outputs, dtype=torch.bool)] = float('-inf')  # Set disallowed outputs to -inf
            

            ##Combine the NN and the Rules output
            #resize the tensors
            # Reshape the tensors to have the same number of rows (elements along the 0th dimension)
            tensor_result = tensor_result.view(-1, 1)
            rule_output = rule_output.view(-1, 1)
            # Concatenate the outputs along dimension 1
            # print(tensor_result.shape)
            # print(rule_output.shape)
            combined_output = torch.cat([tensor_result, rule_output], dim=1)
            # Pass through the linear layer to learn weights
            learned_weights = torch.softmax(self.weights(combined_output), dim=1)
            # Split the learned weights for each model
            learned_weights1 = learned_weights[:, :tensor_result.size(1)]
            learned_weights2 = learned_weights[:, tensor_result.size(1):]
            
            # Apply the learned weights to the model outputs
            weighted_output1 = learned_weights1 * tensor_result
            weighted_output2 = learned_weights2 * rule_output
            
            # Sum the weighted outputs for the final prediction
            final_output = weighted_output1 + weighted_output2
        
            return final_output.t()

    # Initialize the model
    input_size = embeddings.size(dim =1)  # Replace with the actual number of features
    num_amr_roles = len(mapping.keys())
    flat_values = [item for value in mapping.values() for item in (value if isinstance(value, list) else [value])]
    num_classes_output = len(set(flat_values))

    dataset = TensorDataset(embeddings, amr_roles, umr_roles,rule_outputs)

    print("input size", input_size)
    print("num amr roles", num_amr_roles)
    print("num classes_ouptut",num_classes_output)

    model = CustomModel(input_size, num_classes_output, num_amr_roles, mapping)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50

    for epoch in range(num_epochs):
        for x, amr_role, target, rule_output in dataset:
            optimizer.zero_grad()

            # Forward pass
            output = model(x, amr_role, rule_output)

            # Compute loss
            target = target.view(1)
            
            loss = criterion(output, target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    return model


def predict(model, test_data, swap_umr_int_dict, swap_amr_int_dict):
    embeddings,amr_role, umr_role, X, rule_outputs = test_data
    dataset = TensorDataset(embeddings, amr_role, umr_role, rule_outputs)
    with torch.no_grad():
        predictions = []
        model.eval()
        for x, amr_role, umr_role,rule_output in dataset:
        # predict and swap it back from an integer to the class
            predictions.append(swap_umr_int_dict[torch.argmax(model(x, amr_role, rule_output)).item()])
   
    #convert the numbers back to categorical data
    y_preds = pd.Series(predictions, name = "y_pred")

    X['amr_role'] = X['amr_role'].map(swap_amr_int_dict)
    X['umr_role'] = X['umr_role'].map(swap_umr_int_dict)

    print("STARTING HERE /n")
    print(y_preds)
    

    X["y_pred"] = predictions

    return X, model.weights.weight
    
   


# Once trained, you can use the model for predictions
# Replace `your_input_data` with your actual input data and letter


# Convert the predictions to class labels if needed
#predicted_labels = torch.argmax(predictions, dim=1)

# original_categories = df['CategoryColumn'].cat.categories
# mapped_values = other_numerical_tensor.numpy().tolist()
# original_values = pd.Series(mapped_values).map(dict(enumerate(original_categories)))


# The predicted_labels are the predicted classes for your output

def run_nn_with_rules():
    embeddings,amr_role, umr_role, X, mapping, swap_umr_int_dict, swap_amr_int_dict, rule_outputs= preprocessing("train")
    model = train_model(embeddings,amr_role, umr_role ,mapping, rule_outputs)
    df_test = predict(model, (embeddings, amr_role, umr_role,X, rule_outputs), swap_umr_int_dict, swap_amr_int_dict) 

    df_test.to_csv("output/nn_with_rules_test_2.csv")

    #sklearn.metrics.accuracy_score(y_true, predictions)
    #print(type(predictions), type(y_true))


def run_splits_nn(model_choice):
    model_list = ["base_nn", "nn_with_rules_weights", "baseline"]
    if model_choice not in model_list:
        print("pick a model that has been created")
        return
    

    embeddings,amr_role, umr_role, X, mapping, swap_umr_int_dict, swap_amr_int_dict, rule_output = preprocessing("train")
    
    embeddings_1,amr_role_1, umr_role_1, X_1, mapping_1, swap_umr_int_dict,swap_amr_int_dict,rule_outputs_1 = preprocessing("test")
    all_embeddings=  torch.cat((embeddings,embeddings_1), 0)
    all_amr_roles = torch.cat((amr_role,amr_role_1),0)
    all_umr_roles = torch.cat((umr_role, umr_role_1),0)
    all_rule_outputs = torch.cat((rule_output, rule_outputs_1),0)
    all_Xs = pd.concat((X,X_1),axis=0)
    weights = []


    splits= get_indices(all_umr_roles)

    for i, (train_index, test_index) in splits:
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        
        #select training data
        embeddings =  torch.index_select(all_embeddings, 0, torch.LongTensor(train_index))
        amr_roles = torch.index_select(all_amr_roles, 0,torch.LongTensor(train_index) )
        umr_roles = torch.index_select(all_umr_roles, 0,torch.LongTensor(train_index) )
        rule_outputs = torch.index_select(all_rule_outputs, 0,torch.LongTensor(train_index) )


        model = train_model(embeddings,amr_roles, umr_roles,mapping, rule_outputs)

        #select test data
        embeddings =  torch.index_select(all_embeddings, 0, torch.LongTensor(test_index))
        amr_roles = torch.index_select(all_amr_roles, 0,torch.LongTensor(test_index))
        umr_roles = torch.index_select(all_umr_roles, 0,torch.LongTensor(test_index))
        rule_outputs = torch.index_select(all_rule_outputs, 0,torch.LongTensor(test_index) )

        Xs = all_Xs.iloc[test_index.tolist()]

        df_test, weight_i = predict(model, (embeddings, amr_roles, umr_roles,Xs, rule_outputs), swap_umr_int_dict, swap_amr_int_dict) 
        weights.append(weight_i)


        df_test.to_csv(f"output/k-fold/{model_choice}_test_{i}.csv")
    final_weights_folder = "output/k-fold/combined_nn_weights.txt"
    with open(final_weights_folder, 'w') as f:
        for line in weights:
            f.write(f"{line}\n")
    return df_test



if __name__ == "__main__":
    #run_nn_with_rules()
    run_splits_nn("nn_with_rules_weights")
    
