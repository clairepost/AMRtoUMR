#This script will preprocess data from input data and divide it into splits that can be used for PSL

#iterate through these each time to get the results for all of the splits


#IMPORT STATEMENTS
import torch
import pandas as pd
import os
import sys
import os
from error_analysis import get_indices
from nn_with_rules_weights import preprocessing

#IMPORTANT VARIABLES
i = 0  #change i to change which split we are using 0-4
output_path = os.path.join(os.getcwd(), "PSL/data/UMR")


#READ IN DATA AND COMBINE
embeddings,amr_role, umr_role, X, mapping, swap_umr_int_dict, swap_amr_int_dict, rule_output = preprocessing("train")
embeddings_1,amr_role_1, umr_role_1, X_1, mapping_1, swap_umr_int_dict,swap_amr_int_dict,rule_outputs_1 = preprocessing("test")
all_embeddings=  torch.cat((embeddings,embeddings_1), 0)
all_amr_roles = torch.cat((amr_role,amr_role_1),0)
all_umr_roles = torch.cat((umr_role, umr_role_1),0)
all_rule_outputs = torch.cat((rule_output, rule_outputs_1),0)
all_Xs = pd.concat((X,X_1),axis=0)
weights = []

#GET SPLITS 
splits= get_indices(all_umr_roles)
splits = list(splits)
i, (train_index,test_index) = splits[i]


#select training data given split
embeddings =  torch.index_select(all_embeddings, 0, torch.LongTensor(train_index))
amr_roles = torch.index_select(all_amr_roles, 0,torch.LongTensor(train_index) )
umr_roles = torch.index_select(all_umr_roles, 0,torch.LongTensor(train_index) )
rule_outputs = torch.index_select(all_rule_outputs, 0,torch.LongTensor(train_index) )


#select test datagiven split
embeddings =  torch.index_select(all_embeddings, 0, torch.LongTensor(test_index))
amr_roles = torch.index_select(all_amr_roles, 0,torch.LongTensor(test_index))
umr_roles = torch.index_select(all_umr_roles, 0,torch.LongTensor(test_index))
rule_outputs = torch.index_select(all_rule_outputs, 0,torch.LongTensor(test_index))



#OUTPUT DATA TO FILES
# File name
file_name = "example.txt"
# File content
file_content = "This is an example file."# Write content to the new file
# Path to the new file
new_file_path = os.path.join(output_path, file_name)
with open(new_file_path, 'w') as file:
    file.write(file_content)

