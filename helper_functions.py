import pandas as pd
import os
from str2graph import create_graph
import re
import torch
from alignment import *

def remove_comment_lines(input_string):
    # Use regular expression to remove lines starting with #
    result_string = re.sub(r'^\s*#.*$', '', input_string, flags=re.MULTILINE)

    return result_string

def read_training_data(folder):
    amr_graphs = {}
    umr_graphs = {} #keys are file name, values are list of sentences
    for file in os.listdir(folder):
        df = pd.read_csv(folder + "/" + file) #read file

        df.dropna(subset=['UMR'], inplace=True) #remove the rows that don't have umr graphs
        #get AMR, UMR cols
        amr_graphs_str = df["AMR"]
        umr_graphs_str = df["UMR"] 
        sents = df["sentence"].tolist()
        ne_info = df["Named Entity"].tolist()

        #set up dict for this file, dicts are used just for consiticncy sake of the rest of the data
        amr_graphs[file] =[]
        umr_graphs[file] = []

        for i in amr_graphs_str:
            i = remove_comment_lines(i)
            amr_graphs[file].append(create_graph(i))
        
        for i in umr_graphs_str:
            umr_graphs[file].append(create_graph(i))


       
    amr_roles= {
       ":mod",
       ":cause",
       ":part", 
       ":consist-of",
       ":source",
       ":destination",
       ":condition",
       ":ARG1-of"
        } # I think remove concession
    
    amr_roles_in_tail= {
       ":ARG1-of": "cause-01"
    }
    umr_t2r = {
        "cause-01":[":cause", ":reason",":Cause-of"]
    }
        

    splits_data = align_graphs_on_AMR_splits(sents,ne_info ,amr_graphs,umr_graphs,amr_roles,amr_roles_in_tail, umr_t2r)
    splits_data_df = pd.DataFrame(splits_data)

    columns = ["file", "sent_i","sent","ne_info", "amr_graph","amr_head_name", "amr_tail_name", "amr_role","umr_head_name","umr_tail_name", "umr_role", "amr_head_id", "umr_head_id", "amr_tail_id", "umr_tail_id"]

    splits_data_df.columns= columns
    splits_data_df.to_csv('sample_df.csv', index=False)
    return splits_data_df

# def reformat_x(X,bert_embeddings, N,D):
#     #this function flattens the bert embeddings so that the input vecotr can be the appropriate size
#     # Combine all inputs into a list
#     combined_inputs = []
#     X=tf.convert_to_tensor(X)
#     print(bert_embeddings)
#     print(bert_embeddings.size())
#     print(type(X))
#     for i in range(len(X)):
#         combined_input = torch.cat((bert_embeddings.view(-1,N*D),X[i]),dim=1)
#         combined_inputs.append(combined_input)
#     combined_inputs_tensor = torch.cat(combined_inputs, dim=0)
#     return X, embeddings
        


read_training_data("training_data")