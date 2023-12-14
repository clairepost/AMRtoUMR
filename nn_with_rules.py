#This function will create a base NN 
#Steps
#1 : preprocess input data to just be the sentence embedding and one hot encoding of the relation
#2 : create the neural network. The neural network architecture allows for contraining the output depending on the feature role that we provide

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from helper_functions import get_embeddings, create_mapping
import ast

def preprocessing(training):
    #training is a boolean: set it to trtue, to preprocess the training data, and false to preprocess the test data
    #load in data, get bert embeddings, and set it up as tensors

    if training:
        X= pd.read_csv("x_train.csv")
        X['ne_info'] = X['ne_info'].apply(ast.literal_eval)
        y_true= pd.read_csv("y_trues_train.csv") 
    else:
        X= pd.read_csv("x_test.csv")
        X['ne_info'] = X['ne_info'].apply(ast.literal_eval)
        y_true= pd.read_csv("y_trues_test.csv") 

    X = pd.concat([X,y_true],axis = 1)
   
    mapping ,swap_amr_int_dict,swap_umr_int_dict = create_mapping()

    # Convert the categorical column to numerical form using the mapping
    X['amr_role'] = X['amr_role'].map(swap_amr_int_dict)
    X['umr_role'] = X['umr_role'].map(swap_umr_int_dict)
    X = X.dropna(subset=['umr_role','amr_role']).reset_index(drop=True) #remove missing y_true
    umr_role = torch.tensor(X["umr_role"],dtype=torch.long)
    amr_role = torch.tensor(X['amr_role'], dtype=torch.long)
    embeddings = get_embeddings(X) 
    y_rules = detect
    

    #print sizes of returned data
    print("amr_role" , amr_role.size())
    print("umr_role" , umr_role.size())
    print("embeddings size", embeddings.size()) #size ([50,768])

    return embeddings,amr_role, umr_role, X,y_true, mapping, swap_umr_int_dict,swap_amr_int_dict #return X and y_truefor mapping back to the categories later

def train_model(embeddings, amr_role, umr_role,mapping):

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
    num_classes_output = len(set(flat_values))

    print("input size", input_size)
    print("num amr roles", num_amr_roles)
    print("num classes_ouptut",num_classes_output)

    model = CustomModel(input_size, num_classes_output, num_amr_roles, mapping)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 40

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


def predict(model):
    embeddings,amr_role, umr_role, X,y_true, mapping, swap_umr_int_dict,swap_amr_int_dict = preprocessing(False)
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
    print(y_preds,X)
    df = pd.concat([X,y_preds],axis = 1)
    df.to_csv("base_nn_test.csv")
    return predictions, y_true


# Once trained, you can use the model for predictions
# Replace `your_input_data` with your actual input data and letter


# Convert the predictions to class labels if needed
#predicted_labels = torch.argmax(predictions, dim=1)

# original_categories = df['CategoryColumn'].cat.categories
# mapped_values = other_numerical_tensor.numpy().tolist()
# original_values = pd.Series(mapped_values).map(dict(enumerate(original_categories)))


# The predicted_labels are the predicted classes for your output


embeddings,amr_role, umr_role, X,y_true, mapping, swap_umr_int_dict, swap_amr_int_dict= preprocessing(True)
model = train_model(embeddings,amr_role, umr_role,mapping)
predictions,y_true = predict(model)
print(type(predictions), type(y_true))
