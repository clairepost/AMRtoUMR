from helper_functions import read_training_data
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from rules import detect_split_role
from transformers import BertModel, BertTokenizer

def get_embeddings(data):
    # Load pre-trained BERT model and tokenizer
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    embeddings = []
    for i in range(len(data)):
        # Example input text
        text = data["sent"][i]

        # Tokenize input text and get BERT embeddings
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = bert_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)  # Using mean pooling for simplicity
        
        embeddings.append(embedding)
    data["embeddings"] = embeddings
    return data
    # Now, you can use 'embeddings' as input features for your own neural network


def extract_data():
    #creat X's and Y's
    #X's should be of the form [(amr_head_name1,amr_role1,amr_tail_name1),(h2,r2,t2)]
    #Y's will be of the form [umr_role1, umr_role2]
    df = read_training_data("training_data")
    X_columns = ['sent','ne_info' ,'amr_graph','amr_head_name', 'amr_role', 'amr_tail_name']
    Y_columns = ['umr_role']
    # Create a new DataFrame X with the selected columns
    X = df[X_columns].copy()
    Y = df[Y_columns].copy()
    # If you want X to be a list of tuples, you can use the to_records() method
    X_tuples = list(X.to_records(index=False))

    # 1st push - format
    # input: [sent, G, h, r,t],..., ...] of length n

    # 2nd push - access to cause-01: (h, :arg1-of, cause-01)

    # output: ["cause", "reason", .....] of length n (1 choice)
    # output: [(["cause", "reason"], [.75,.25]), (["mod"],[1]), ....] - initial approach

    # Example output
    return X_tuples, Y

def data_programming(X,Y):

    
    #TO DO: make sure the input is in the format that Benet expects
    Y_rules = [Y]
    f_l = []
    #Y_rules = detect_split_role(X,Y) just using place holder rules for right now
    for l_x in Y_rules:
        rule_f = (l_x == Y).astype(int)
        f_l.append(rule_f)
    return f_l
    

def train_model(X,Y):
    # Define your virtual evidence model, neural network model, and data
    class VirtualEvidenceModel(nn.Module):
        def forward(self, X, Y):
            # Implement logic for the virtual evidence model
            exp(wv · fv(X, Y ))
            

    class NeuralNetworkModel(nn.Module):
        def forward(self, X, Y):
            # Implement logic for the neural network model
            pass

    # Initialize weights w_v for each v and set up K
    rules = data_programming(X,Y)
    num_weights = len(rules)
    phi_0 = [nn.Parameter(torch.randn(1, requires_grad=True)) for _ in range(num_weights)] #initializing random weights for Virtual Evidence
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
    X =  pd.DataFrame.from_records(X, columns = ['sent','ne_info' ,'amr_graph','amr_head_name', 'amr_role', 'amr_tail_name'])
    X = get_embeddings(X)
    print(data_programming(X,Y))
    trained_model = train_model(X,Y)
    #test_data_x, test_data_y = extract_data()
    #y_preds = train_model.predict(test_data)
    #compare y_preds to y_test
