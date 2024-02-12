#This script will preprocess data from input data and divide it into splits that can be used for PSL

#IMPORT STATEMENTS
import torch
import pandas as pd
import os
import numpy as np
import sys
import os
from error_analysis import get_indices
from nn_with_rules_weights import preprocessing

#IMPORTANT VARIABLES and HELPER FUNCTION
i = 0  #change i to change which split we are using 0-4
output_path = os.path.join(os.getcwd(), "PSL/data/UMR")


def save_to_file(tensor, indices, file_name, target=False):
# Convert tensor to NumPy array and saves it
        array = tensor.numpy().astype(int)
        print(array)
        indices_column = indices.numpy().astype(int)  # Ensure indices is 2D
        print(indices_column)   

        if target:
                #If this is the target version, then we need to add a column of all 1's to indicate truth
                ones_array = np.ones_like(array)
                array_with_indices = np.column_stack((indices_column,array, ones_array))
        else:
                array_with_indices = np.column_stack((indices_column,array))


        # Path to the new file
        new_file_path = os.path.join(output_path, file_name)



        # Save NumPy array as tab-separated text file
        np.savetxt(new_file_path, array_with_indices,fmt='%d', delimiter='\t')



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
i, (train_index,test_index) = splits[i]\

train_indices = torch.LongTensor(train_index)
test_indices = torch.LongTensor(test_index)


#select training data given split
#embeddings =  torch.index_select(all_embeddings, 0, torch.LongTensor(train_index))
amr_roles = torch.index_select(all_amr_roles, 0,train_indices )
save_to_file(amr_roles, train_indices, "amr_obs_learn.txt")
umr_roles = torch.index_select(all_umr_roles, 0,train_indices )
save_to_file(umr_roles, train_indices, "umr_target_learn.txt")
save_to_file(umr_roles, train_indices, "umr_truth_learn.txt", True)
#rule_outputs = torch.index_select(all_rule_outputs, 0,torch.LongTensor(train_index) )


#select test datagiven split
#embeddings =  torch.index_select(all_embeddings, 0, torch.LongTensor(test_index))
amr_roles = torch.index_select(all_amr_roles, 0,test_indices)
save_to_file(amr_roles, test_indices, "amr_obs_eval.txt")
umr_roles = torch.index_select(all_umr_roles, 0,test_indices)
save_to_file(umr_roles, test_indices, "umr_target_eval.txt")
save_to_file(umr_roles, test_indices, "umr_truth_eval.txt", True)
#rule_outputs = torch.index_select(all_rule_outputs, 0,torch.LongTensor(test_index))








