import pandas as pd
import os
from str2graph import create_graph
import re
import torch
from sklearn.preprocessing import LabelEncoder
from alignment import *
from sklearn.preprocessing import LabelEncoder
from animacyParser import parse_animacy_runner

def extract_data(training):
    #training parameter: boolean- True if you want to extract training, false if you want to extract test
    #creat X's and Y's
    #X's should be of the form [(amr_head_name1,amr_role1,amr_tail_name1),(h2,r2,t2)]
    #Y's will be of the form [umr_role1, umr_role2]
    if training:
        df = read_training_data("training_data")
    else:
        df = read_test_data()
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
        
def read_test_data():
    #THIS FUNCTION IS MOSTLY COPIED OVER FROM FINALY_PROJECT.IPYNB
    # put all files in dicts
    umr_files = {}
    amr_files = {}

    umr_path = os.getcwd() + '/UMR-data-english'
    amr_path = os.getcwd() + '/AMR-data-english'

    for f in os.listdir(umr_path):
        file1 = open(umr_path + '/' + f, 'r')
        umr_files[f] = file1.read()

    for f in os.listdir(amr_path):
        file1 = open(amr_path + '/' + f, 'r')
        amr_files[f] = file1.read()

    #Do not have all of the UMR annotations for the Putin document, so just grab everything before this sentence id from AMR
    amr_files["putin_ENG_0152_2000_1208-AMR.txt"] = amr_files["putin_ENG_0152_2000_1208-AMR.txt"].split("::id NW_PRI_ENG_0152_2000_1208.13")[0]

    #create file mappings
    #Change key-names to the mapping above, allows for easier comparison
    amr_files[1] = amr_files.pop('lindsay-AMR.txt')
    amr_files[2] = amr_files.pop('lorpt-024_Phillipines_landslide_AMR.txt')
    amr_files[3] = amr_files.pop('putin_ENG_0152_2000_1208-AMR.txt')
    amr_files[4] = amr_files.pop('edmund_pope-AMR.txt')
    amr_files[5] = amr_files.pop('pear-AMR__of__english-umr-0004.txt')

    umr_files[1] = umr_files.pop('lindsay-umr.txt')
    umr_files[2] = umr_files.pop('Lorelei_lorpt-024_Philippines_Landslide_2023-release.txt')
    umr_files[3] = umr_files.pop('lorelei_lorpt-151_putin_2023-release.txt')
    umr_files[4] = umr_files.pop('lorelei_lorpt-152_edmundpope_2023-release.txt')
    umr_files[5] = umr_files.pop('Pear_Story_2023-release.txt')

    file_map = {1:"Lindsay",2: "Landslide", 3:"Putin",4:"Edmund Pope", 5:"Pear Story"}

    umr_sents = {}
    all_sentences = []
    for f in umr_files:
        umr_sents[f] = re.findall(r'(?<=sentence level graph:\n)\([^#]*(?=\n\n#)', umr_files[f])

    amr_sents = {}
    for f in amr_files:
        amr_sents[f] = re.findall(r'(?<=[\n])\([^#]*(?=\n|)', amr_files[f])
        sentences = re.findall(r'(?<=# ::snt\s).+?(?=\n)',amr_files[f]) #first look
        if not sentences:
            sentences = re.findall(r'(?<=:: snt)[^\n:]*(?=\n)',amr_files[f])#second look
            sentences = [re.sub(r'^\d+\s*', '', element) for element in sentences]
        all_sentences.extend(sentences)


    #using the str2graph.create_graph() function
    umr_graphs = {}
    for file in umr_sents.keys():
        umr_graphs[file] = []
        for sent in umr_sents[file]:
            umr_graphs[file].append(create_graph(sent))

    amr_graphs = {}
    for file in amr_sents.keys():
        amr_graphs[file] = []
        for sent in amr_sents[file]:
            amr_graphs[file].append(create_graph(sent))

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
    ne_info = parse_animacy_runner(all_sentences)
    splits_data = align_graphs_on_AMR_splits(all_sentences,ne_info, amr_graphs,umr_graphs,amr_roles, amr_roles_in_tail, umr_t2r)
    splits_data_df = pd.DataFrame(splits_data)

    columns = ["file", "sent_i","sent","ne_info", "amr_graph","amr_head_name", "amr_tail_name", "amr_role","umr_head_name","umr_tail_name", "umr_role", "amr_head_id", "umr_head_id", "amr_tail_id", "umr_tail_id"]

    splits_data_df.columns= columns
    return splits_data_df

def map_categorical_to_tensor(series):
    """
    Map categorical data in a pandas Series to a PyTorch tensor of numerical values using LabelEncoder.

    Parameters:
    - series (pandas Series): The categorical data in a pandas Series to be mapped.

    Returns:
    - torch.Tensor: The numerical representation of the input categorical data as a PyTorch tensor.
    """
    label_encoder = LabelEncoder()
    numerical_data = label_encoder.fit_transform(series)
    numerical_tensor = torch.tensor(numerical_data, dtype=torch.long)
    return numerical_tensor




def create_mapping_dict(series):
    """
    Create a dictionary mapping unique values in a pandas Series to numerical values using LabelEncoder.

    Parameters:
    - series (pandas Series): The categorical data in a pandas Series.

    Returns:
    - dict: A dictionary mapping unique values to their corresponding numerical representations.
    """
    label_encoder = LabelEncoder()
    numerical_data = label_encoder.fit_transform(series)
    
    mapping_dict = dict(zip(series.unique(), numerical_data))
    return mapping_dict

def map_categorical_to_tensor(series, mapping_dict):
    """
    Map categorical data in a pandas Series to a PyTorch tensor of numerical values using a pre-defined mapping dictionary.

    Parameters:
    - series (pandas Series): The categorical data in a pandas Series.
    - mapping_dict (dict): The mapping dictionary created using create_mapping_dict.

    Returns:
    - torch.Tensor: The numerical representation of the input categorical data as a PyTorch tensor.
    """
    numerical_data = series.map(mapping_dict)
    numerical_tensor = torch.tensor(numerical_data, dtype=torch.long)
    return numerical_tensor

