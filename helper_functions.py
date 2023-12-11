import pandas as pd
import os
from str2graph import create_graph
import re
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
        

    splits_data = align_graphs_on_AMR_splits(sents, amr_graphs,umr_graphs,amr_roles,amr_roles_in_tail, umr_t2r)
    splits_data_df = pd.DataFrame(splits_data)

    columns = ["file", "sent_i","sent", "amr_graph","amr_head_name", "amr_tail_name", "amr_role","umr_head_name","umr_tail_name", "umr_role", "amr_head_id", "umr_head_id", "amr_tail_id", "umr_tail_id"]

    splits_data_df.columns= columns
    splits_data_df.to_csv('sample_df.csv', index=False)
    return splits_data_df

        


read_training_data("training_data")