from helper_functions import read_training_data
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from rules import detect_split_role
from transformers import BertModel, BertTokenizer, BertConfig

def get_embeddings(data):
    # Load pre-trained BERT model and tokenizer and config info
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    config =  BertConfig.from_pretrained("bert-base-uncased")
    N_BERT = config.num_hidden_layers
    D_BERT = config.hidden_size
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
    b = torch.Tensor(len(data), N_BERT,D_BERT)

    return torch.cat(embeddings, out = b), N_BERT, D_BERT



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

    return X_tuples, Y["umr_role"]

def data_programming(X,Y):    
    #TO DO: make sure the input is in the format that Benet expects
    Y_rules = [Y]
    f_l = []
    #Y_rules = detect_split_role(X,Y) just using place holder rules for right now
    for l_x in Y_rules:
        rule_f = (l_x == Y)
        f_l.append(rule_f)
    return torch.tensor(f_l,dtype=float)
    


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
            output = torch.relu(K)

            return output

    class NeuralNetworkModule(nn.Module):
        def __init__(self, input_size, output_size):
            super(NeuralNetworkModule, self).__init__()
            # Replace with your actual neural network initialization
            self.fc = nn.Linear(input_size, output_size)
            self.double()

        def forward(self, X):
            # Replace with your actual neural network forward pass
            return self.fc(X)

    class DPLModel(nn.Module):
        def __init__(self, input_size, latent_variable_size, output_size):
            super(DPLModel, self).__init__()
            self.probabilistic_logic = ProbabilisticLogicModule(input_size, latent_variable_size)
            self.neural_network = NeuralNetworkModule(latent_variable_size, output_size)
            self.double()

        def forward(self, X):
            # Forward pass through probabilistic logic
            latent_variables = self.probabilistic_logic(X)

            # Forward pass through neural network
            predictions = self.neural_network(latent_variables)

            return predictions

    embeddings,N,D= get_embeddings(X)
    f_x_y = data_programming(X,Y)
    input_size = len(X)

    print("input_size", type(X))
    print("latent_var_size", f_x_y.size())
    print("output_size", type(Y), len(set(Y)))
    latent_variable_size = len(f_x_y) #number of rules
    output_size = len(set(Y))


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

        # Compute negative log-likelihood (replace with your actual likelihood function)
        negative_log_likelihood = -compute_likelihood(K, predictions)

        # Zero the gradients
        optimizer.zero_grad()

        # Backpropagation
        negative_log_likelihood.backward()

        # Update parameters
        optimizer.step()

        # Optionally print or log the loss for monitoring
        if step % 100 == 0:
            print(f"Step {step}, Loss: {negative_log_likelihood.item()}")

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
    X,Y= extract_data()
    X =  pd.DataFrame.from_records(X, columns = ['sent','ne_info' ,'amr_graph','amr_head_name', 'amr_role', 'amr_tail_name'])
    print(data_programming(X,Y))
    trained_model = train_model(X,Y)
    #test_data_x, test_data_y = extract_data()
    #y_preds = train_model.predict(test_data)
    #compare y_preds to y_test
