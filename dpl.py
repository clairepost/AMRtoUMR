from helper_functions import extract_data, create_mapping_dict, map_categorical_to_tensor, get_embeddings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from rules import detect_split_role



def data_programming(X,Y):    
    #TO DO: make sure the input is in the format that Benet expects
    f_l = []
    Y_rules =[detect_split_role(X)] #just using place holder rules for right now

    for l_x in Y_rules:

        rule = []
        for i in range(len(l_x)):
            label = l_x[i][0]
            weight = l_x[i][1]
 
            if Y[i] in label:
                score = weight[label.index(Y[i])]
            else:
                score = 0

            rule.append(score)
            
        f_l.append(rule)
    print(f_l)
    f_l =  torch.tensor(f_l,dtype=float)
    print("size of rules", f_l.size())
    return f_l
    


def train_model(X,Y):
    class ProbabilisticLogicModule(nn.Module):
        def __init__(self, input_size, latent_variable_size):
            super(ProbabilisticLogicModule, self).__init__()
            # Initialize custom weights
            self.weight = nn.Parameter(torch.randn(latent_variable_size, 1))
            print(self.weight.size())
            self.F_X_Y = data_programming(X,Y)
            self.double()
            

        def forward(self, X):
            # Perform a custom operation using the parameters
            K = torch.exp(torch.matmul(self.weight.t(),self.F_X_Y))

            # Apply an activation function (e.g., ReLU)
            #output = torch.relu(K)

            return K

    class NeuralNetworkModule(nn.Module):
        def __init__(self, input_size, latent_size, output_size):
            hidden_size = 64
            super(NeuralNetworkModule, self).__init__()
            # Replace with your actual neural network initialization
            self.fc1 = nn.Linear(input_size + latent_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.double()

        def forward(self, X,K):
            # Replace with your actual neural network forward pass
            print(X.size())
            print(K.size())
            x_with_k = torch.cat([X, K], dim=0)
            print(x_with_k.size())
            # Apply the factor Phi(X, K)
            x_with_k = self.fc1(x_with_k)
            x_with_k = self.relu(x_with_k)
            psi_result = self.fc2(x_with_k)
            return psi_result

    class DPLModel(nn.Module):
        def __init__(self, input_size, latent_variable_size, output_size):
            super(DPLModel, self).__init__()
            self.probabilistic_logic = ProbabilisticLogicModule(input_size, latent_variable_size)
            self.neural_network = NeuralNetworkModule(input_size, latent_variable_size, output_size)
            self.double()

        def forward(self, X):
            # Forward pass through probabilistic logic
            latent_variables = self.probabilistic_logic(X)

            # Forward pass through neural network
            predictions = self.neural_network(latent_variables,latent_variables)

            return predictions

    #Preprocessing and Setup
    embeddings,N,D= get_embeddings(X)
    f_x_y = data_programming(X,Y)
    
    mapping_dict=create_mapping_dict(Y)
    Y_tensor = map_categorical_to_tensor(Y, mapping_dict)

    print("input_size", type(X))
    print("latent_var_size", f_x_y.size())
    print("output_size", type(Y), len(set(Y)))
    latent_variable_size = len(f_x_y) #number of rules
    output_size = len(set(Y))
    input_size = len(X)


    # Instantiate DPL model
    dpl_model = DPLModel(input_size, latent_variable_size, output_size)

    # Set up optimizer
    optimizer = torch.optim.SGD(dpl_model.parameters(), lr=0.01)


    # Number of optimization steps
    num_steps = 1000

    # Optimization loop
    for step in range(num_steps):
        # Forward pass through the DPL model
        predictions = dpl_model(X)


        # Compute the conditional likelihood (softmax)
        p_k_given_x = torch.nn.functional.softmax(predictions, dim=1)

        # Compute the negative log-likelihood
        loss = -torch.log(p_k_given_x[range(num_samples), Y]).mean()


        # Zero the gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Optionally print or log the loss for monitoring
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

# After the optimization loop, your DPL model parameters are trained

# def train_model(X,Y):
#     # Define your virtual evidence model, neural network model, and data
#     class VirtualEvidenceModel(nn.Module):
#         def forward(self, X, Y):
#             # Implement logic for the virtual evidence model
#             #return exp(wv * fv(X, Y ))
#             pass
            
#     class NeuralNetworkModel(nn.Module):
#         def __init__(self, input_size, output_size):
#             super(NeuralNetworkModel, self.__init__())
#             self.fc = nn.Linear(input_size, output_size)
        
#         def forward(self, X, Y):
#             #NN takes in X, the tensor
#             # Implement logic for the neural network model
#             return torch.softmax(self.fc(X),dim = 1)

#     #add BERT embeddings to X
#     embeddings,N,D= get_embeddings(X)

#     input_size = embeddings.size(-1)
    
#     # Initialize weights w_v for each v and set up K
#     rules = data_programming(X,Y)
#     num_weights = len(rules)
#     phi_0 = [nn.Parameter(torch.randn(1, requires_grad=True)) for _ in range(num_weights)] #initializing random weights for Virtual Evidence
#     K = [np.exp(phi_0 * [v] * rules[v]) for v in range(len(rules))]


#     # Initialize virtual evidence model, neural network model, and parameters
#     virtual_evidence_model = VirtualEvidenceModel()
#     neural_network_model = NeuralNetworkModel(input_size,output_size)
#     optimizer_phi = optim.SGD(virtual_evidence_model.parameters(), lr=0.001)
#     optimizer_psi = optim.SGD(neural_network_model.parameters(), lr=0.001)

    
#     # Number of iterations
#     T = 100

#     # E-step and M-step
#     for t in range(1, T + 1):
#         # E-step: Compute variational approximation q(Y)
#         q_Y = virtual_evidence_model(X, Y)

#         # M-step: Update virtual evidence model parameters (Φ) and neural network model parameters (Ψ)
#         optimizer_phi.zero_grad()
#         optimizer_psi.zero_grad()

#         # Compute KL divergence for Φ
#         kl_phi = torch.distributions.kl.kl_divergence(q_Y, virtual_evidence_model(X, Y))
#         kl_phi.backward()
#         optimizer_phi.step()

#         # Compute KL divergence for Ψ
#         kl_psi = torch.distributions.kl.kl_divergence(q_Y, neural_network_model(X, Y))
#         kl_psi.backward()
#         optimizer_psi.step()

#     # Return the learned neural network model
#     learned_neural_network = neural_network_model
#     return learned_neural_network


if __name__ == "__main__":
    X,Y= extract_data(True) # gets training data
    X =  pd.DataFrame.from_records(X, columns = ['sent','ne_info' ,'amr_graph','amr_head_name', 'amr_role', 'amr_tail_name'])
    print(data_programming(X,Y))
    trained_model = train_model(X,Y)
    #test_data_x, test_data_y = extract_data()
    #y_preds = train_model.predict(test_data)
    #compare y_preds to y_test
