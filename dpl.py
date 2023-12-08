from helper_functions import read_training_data
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rules import detect_split_role

def extract_data():
    #creat X's and Y's
    #X's should be of the form [(amr_head_name1,amr_role1,amr_tail_name1),(h2,r2,t2)]
    #Y's will be of the form [umr_role1, umr_role2]
    df = read_training_data("training_data")
    X_columns = ['amr_head_name', 'amr_role', 'amr_tail_name']
    Y_columns = ['umr_role']
    # Create a new DataFrame X with the selected columns
    X = df[X_columns].copy()
    Y = df[Y_columns].copy()
    # If you want X to be a list of tuples, you can use the to_records() method
    X_tuples = list(X.to_records(index=False))

    # Example output
    print(len(X_tuples))
    print(len(Y))
    return X_tuples, Y

def data_programming(X,Y):

    num_rules = 2 #change as needed for more rules to be added, num_rules is referred to as v in the paper
    
    #TO DO: make sure the input is in the format that Benet expects
    detect_split_role(X)
    f_l1 = (l1_x == Y).astype(int)

    l2_x = rule_2(x)
    f_l2 = (l1_x == Y).astype(int)

    return [fl_1,fl_2]
    


def train_model(X,Y):
    # Define your virtual evidence model, neural network model, and data
    class VirtualEvidenceModel(nn.Module):
        def forward(self, X, Y):
            # Implement logic for the virtual evidence model
            pass

    class NeuralNetworkModel(nn.Module):
        def forward(self, X, Y):
            # Implement logic for the neural network model
            pass

    # Initialize weights w_v for each v and set up K
    rules = data_programming(X,Y)
    num_weights = len(rules)
    weights = [nn.Parameter(torch.randn(1, requires_grad=True)) for _ in range(num_weights)]
    K = [np.exp(w[v] * rules[v]) for v in range(len(rules))]


    # Initialize virtual evidence model, neural network model, and parameters
    virtual_evidence_model = VirtualEvidenceModel()
    neural_network_model = NeuralNetworkModel()
    optimizer_phi = optim.SGD(virtual_evidence_model.parameters(), lr=0.001)
    optimizer_psi = optim.SGD(neural_network_model.parameters(), lr=0.001)

    
    # Number of iterations
    T = 100

    # E-step and M-step
    for t in range(1, T + 1):
        # E-step: Compute variational approximation q(Y)
        q_Y = virtual_evidence_model(X, Y)

        # M-step: Update virtual evidence model parameters (Φ) and neural network model parameters (Ψ)
        optimizer_phi.zero_grad()
        optimizer_psi.zero_grad()

        # Compute KL divergence for Φ
        kl_phi = torch.distributions.kl.kl_divergence(q_Y, virtual_evidence_model(X, Y))
        kl_phi.backward()
        optimizer_phi.step()

        # Compute KL divergence for Ψ
        kl_psi = torch.distributions.kl.kl_divergence(q_Y, neural_network_model(X, Y))
        kl_psi.backward()
        optimizer_psi.step()

    # Return the learned neural network model
    learned_neural_network = neural_network_model
    return learned_neural_network


if __name__ == "__main__":
    X,Y= extract_data()
    trained_model = train_model(X,Y)
    #test_data_x, test_data_y = extract_data()
    #y_preds = train_model.predict(test_data)
    #compare y_preds to y_test
