import pandas as pd
import os
from str2graph import create_graph
import re
import torch
from sklearn.preprocessing import LabelEncoder
from str2graph import create_graph,draw_graph
from alignment import *
from sklearn.preprocessing import LabelEncoder
from transformers import BertModel, BertTokenizer, BertConfig
from animacyParser import parse_animacy_runner
from rules import detect_split_role

def extract_data(split):
    #training parameter: boolean- True if you want to extract training, false if you want to extract test
    #creat X's and Y's
    #X's should be of the form [(amr_head_name1,amr_role1,amr_tail_name1),(h2,r2,t2)]
    #Y's will be of the form [umr_role1, umr_role2]
    if "train":
        df = read_training_data()
    elif split == "test":
        df = read_test_data()
    elif split == "augment":
        print("reading in the augment sample data")
        df = read_augment_fake_parallel_data()
    else:
        print("extract_data() takes in 1 arg: either 'train' or 'test'")
        return
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

def read_training_data():
    #reads in the raw training data, returns a df consisting of the parsed and aligned graphs
    amr_graphs = {}
    umr_graphs = {} #keys are file name, values are list of sentences
    sents= {}
    ne_info = {}
    folder = "raw_data/training_data"
    for file in os.listdir(folder):
        df = pd.read_csv(folder + "/" + file) #read file
        df.dropna(subset=['UMR'], inplace=True) #remove the rows that don't have umr graphs
        #get AMR, UMR cols
        amr_graphs_str = df["AMR"]
        umr_graphs_str = df["UMR"] 
        sents[file]= df["sentence"].tolist()
        ne_info[file]=  df["Named Entity"].tolist()

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
    splits_data_df.to_csv("input_data/train_data.csv")
    return splits_data_df
    

# def read_augment_data():
#     # reads in the augmented data
#     #reads in the raw training data, returns a df consisting of the parsed and aligned graphs
#     # creates dummy variables for the extra umr columns

#     # step 1 - open files
#     # put all files in dicts
#     augment_files = {}

#     augment_path = os.path.join(os.getcwd(), 'raw_data', 'augment-data-english')

#     for f in os.listdir(augment_path):
#         file_path = os.path.join(augment_path, f)
#         with open(file_path, 'rb') as file1:
#             try:
#                 content = file1.read().decode('utf-8')
#             except UnicodeDecodeError:
#                 # If 'utf-8' decoding fails, try another encoding
#                 content = file1.read().decode('latin-1')
#             augment_files[f] = content

#     # step 2 - get file names
#     #create file mappings augment data
#     augment_files[1] = augment_files.pop('amr-release-3.0-amrs-bolt.txt')
#     augment_files[2] = augment_files.pop('amr-release-3.0-amrs-cctv.txt')
#     augment_files[3] = augment_files.pop('amr-release-3.0-amrs-consensus.txt')
#     augment_files[4] = augment_files.pop('amr-release-3.0-amrs-dfa.txt')
#     augment_files[5] = augment_files.pop('amr-release-3.0-amrs-dfb.txt')
#     augment_files[6] = augment_files.pop('amr-release-3.0-amrs-fables.txt')
#     augment_files[7] = augment_files.pop('amr-release-3.0-amrs-guidelines.txt')
#     augment_files[8] = augment_files.pop('amr-release-3.0-amrs-mt09sdl.txt')
#     augment_files[9] = augment_files.pop('amr-release-3.0-amrs-proxy.txt')
#     augment_files[10] = augment_files.pop('amr-release-3.0-amrs-wb.txt')
#     augment_files[11] = augment_files.pop('amr-release-3.0-amrs-wiki.txt')
#     augment_files[12] = augment_files.pop('amr-release-3.0-amrs-xinhua.txt')

#     file_map_a = {1:"bolt",2:"cctv",3:"consensus",4:"dfa",5:"dfb",6:"fables",7:"guidelines",8:"mt09",9:"proxy",10:"wb",11:"wiki",12:"xinhua"}


#     # step 3 - extract graphs and sentences

#     all_sentences_augment = {}
#     augment_sents = {}
#     for f in augment_files:
#         if f == '.DS_Store':
#             continue  # Skip system files
#         augment_sents[f] = re.findall(r'(?<=[\n])\([^#]*(?=\n|)', augment_files[f])#extract the graphs
#         sentences = re.findall(r'(?<=# ::snt\s).+?(?=\n)', augment_files[f]) #first look
#         if not sentences:
#             sentences = re.findall(r'(?<=:: snt)[^\n:]*(?=\n)',augment_files[f])#second look
#             sentences = [re.sub(r'^\d+\s*', '', element) for element in sentences]
#         all_sentences_augment[f] = sentences
        

#     # step 4 - make graphs
        
#     #using the str2graph.create_graph() function
#     augment_graphs = {}
#     for file in augment_sents.keys():
#         augment_graphs[file] = []
#         for sent in augment_sents[file]:
#             augment_graphs[file].append(create_graph(sent))


#     # step 5 - split role definitions

#     amr_roles= {
#        ":mod",
#        ":cause",
#        ":part", 
#        ":consist-of",
#        ":source",
#        ":destination",
#        ":condition",
#        ":ARG1-of"
#         } # I think remove concession
    
#     amr_roles_in_tail= {
#        ":ARG1-of": "cause-01"
#     }
#     umr_t2r = {
#         "cause-01":[":cause", ":reason",":Cause-of"]
#     }

#     # step 6 - animacy parsing 

#     ne_info = {}
#     for f in all_sentences_augment:
#         ne_info[f] = parse_animacy_runner(all_sentences_augment[f])


#     # step 7 - make a data frame (INCOMPLETE: NEED HELP HERE OUT)
    
#     # define empty umrs 
#     umr_graphs = None

#     splits_data = align_graphs_on_AMR_splits(all_sentences_augment,ne_info, augment_graphs,umr_graphs=None, amr_roles=amr_roles, amr_roles_in_tail=amr_roles_in_tail, umr_t2r=umr_t2r)

#     # fill in the UMR columns with dummy variable of nothing should be fine since it is just predicting, run it through 

#     # Convert splits_data to DataFrame
#     columns = ["file", "sent_i", "sent", "ne_info", "amr_graph", "amr_head_name", "amr_tail_name", "amr_role",
#                "umr_head_name", "umr_tail_name", "umr_role", "amr_head_id", "umr_head_id", "amr_tail_id",
#                "umr_tail_id"]
#     splits_data_df = pd.DataFrame(splits_data, columns=columns)

#     # step 8 - save and return data frame

#     splits_data_df.to_csv("input_data/augment_data.csv")
#     return splits_data_df

def read_augment_fake_parallel_data():
    #reads in the raw training data, returns a df consisting of the parsed and aligned graphs
    # THIS FUNCTION IS MOSTLY COPIED OVER FROM FINALY_PROJECT.IPYNB
    # put all files in dicts
    umr_files = {}
    amr_files = {}

    # umr_path = os.getcwd() + os.path.join(os.getcwd(), 'raw_data', 'umr-fake')
    # amr_path = os.getcwd() + os.path.join(os.getcwd(), 'raw_data', 'augment-sample-data-english2')
    umr_path = os.getcwd() + '/raw_data/umr-fake2'
    amr_path = os.getcwd() + '/raw_data/augment-sample-data-english2'

    
    for f in os.listdir(umr_path):
        file_path = os.path.join(umr_path, f)
        with open(file_path, 'rb') as file1:
            try:
                content = file1.read().decode('utf-8')
            except UnicodeDecodeError:
                # If 'utf-8' decoding fails, try another encoding
                content = file1.read().decode('latin-1')
            umr_files[f] = content

    
    for f in os.listdir(amr_path):
        file_path = os.path.join(amr_path, f)
        with open(file_path, 'rb') as file1:
            try:
                content = file1.read().decode('utf-8')
            except UnicodeDecodeError:
                # If 'utf-8' decoding fails, try another encoding
                content = file1.read().decode('latin-1')
            amr_files[f] = content

    # step 2 - get file names
    #create file mappings augment data
    umr_files[0] = umr_files.pop('umr-fake-consensus.txt')
    umr_files[1] = umr_files.pop('umr-fake-wiki.txt')
    amr_files[0] = amr_files.pop('amr-release-3.0-amrs-consensus.txt')
    amr_files[1] = amr_files.pop('amr-release-3.0-amrs-wiki.txt')

    # file_map = {1:"consensus"}
    file_map = {0:"consensus",1:"wiki"}

    # step 3 - extract graphs and sentences

    umr_sents = {}
    all_sentences = {}
    for f in umr_files:
        # umr_sents[f] = re.findall(r'(?<=sentence level graph:\n)\([^#]*(?=\n\n#)', umr_files[f])
        umr_sents[f] = re.findall(r'(?<=[\n])\([^#]*(?=\n|)', umr_files[f])
        sentences = re.findall(r'(?<=# ::snt\s).+?(?=\n)',umr_files[f]) #first look
        if not sentences:
            sentences = re.findall(r'(?<=:: snt)[^\n:]*(?=\n)',umr_files[f])#second look
            sentences = [re.sub(r'^\d+\s*', '', element) for element in sentences]
        
    # TODO: pass in AMR sents to the X_tuple, add a column with that info, pass said x_tuple directly into the animacy_runner, change that function and call, do animacy parsing all over there (no double calls to animacy)
    amr_sents = {}
    for f in amr_files:
        amr_sents[f] = re.findall(r'(?<=[\n])\([^#]*(?=\n|)', amr_files[f])
        sentences = re.findall(r'(?<=# ::snt\s).+?(?=\n)',amr_files[f]) #first look
        if not sentences:
            sentences = re.findall(r'(?<=:: snt)[^\n:]*(?=\n)',amr_files[f])#second look
            sentences = [re.sub(r'^\d+\s*', '', element) for element in sentences]
        all_sentences[f] = sentences


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
    
    # ne_info = {}
    # for f in all_sentences:
    #     print("AMR sents at F: ", "\n", amr_sents[f])
    #     print("\n\n")
    #     ne_info[f] = parse_animacy_runner(all_sentences[f], amr_sents[f]) # changed this line to send the amr graph with the animacy info 
    
    # GAMEPLAN: send 1 version of splits data for processing, then make another version that processes it into the data set like it should be processed in 

    # sending to get the splits
    print("SENDING TO GET THE SPLITS")
    # splits_data = align_graphs_on_AMR_splits(all_sentences,ne_info, amr_sents,amr_graphs,umr_graphs,amr_roles, amr_roles_in_tail, umr_t2r) # changing this data structure to get the animacy info 
    splits_data = align_graphs_no_ne(all_sentences, amr_sents,amr_graphs,umr_graphs,amr_roles, amr_roles_in_tail, umr_t2r) # changing this data structure to get the animacy info 

    # Convert splits_data to a DataFrame
    splits_data_df_temp = pd.DataFrame(splits_data)

    columnsTemp = ["file", "sent_i","sent","amr_prints", "amr_graph","amr_head_name", "amr_tail_name", "amr_role","umr_head_name","umr_tail_name", "umr_role", "amr_head_id", "umr_head_id", "amr_tail_id", "umr_tail_id"] 

    splits_data_df_temp.columns= columnsTemp

    ne_info = {}
    for f in all_sentences:
        ne_info[f] = parse_animacy_runner(all_sentences[f], amr_sents[f], splits_data_df_temp) 
    

    # ne_info = {}
    # for f in all_sentences:
    #     # what I need to run ne_animacy(named_entity, tail, amr_graph, sentence) from the rules function & fixCause(amr_graph, tail, role)
    #     # named_entity = x_tuple["ne_info"]
    #     # amr_graph = x_tuple["amr_graph"]
    #     #  role = x_tuple["amr_role"]
    #     # tail = x_tuple["amr_tail_name"]
    #     # sentence = x_tuple["sent"]

    #     ne_info[f] = parse_animacy_runner(all_sentences[f], amr_sents[f], amr_graphs[f], splits_data[f]['amr_tail_name'], splits_data[f]['amr_role']) # changed this line to send the amr graph with the animacy info # taking a guess that for this other info we are going to need to just look at index f behind it (might be wrong)
    # Print the contents and length of splits_data
    # print("Contents of splits_data:")
    # for entry in splits_data:
    #     print(entry)

    # print("Length of splits_data:", len(splits_data))
    # print("Length of all_sentences:", len(all_sentences))
    # print("Length of amr_sents:", len(amr_sents))
    # print("Length of amr_graphs:", len(amr_graphs))
    # print("Length of splits_data:", len(splits_data))
    # ne_info = {}
    # for entry in splits_data:
    #     file_idx = entry[0]  # Assuming the file index is at index 0 in each entry
    #     amr_tail_name = entry[7]  # Assuming amr_tail_name is at index 7
    #     amr_role = entry[8]  # Assuming amr_role is at index 8
    #     ne_info[file_idx] = parse_animacy_runner(all_sentences[file_idx], amr_sents[file_idx], amr_graphs[file_idx], amr_tail_name, amr_role)

    # maybe instead I can append a column on after doing adding the ne_animacy info since that is all worked out for me in the align_graph_on_AMR_splits() function (so I would need to delete ne_info and apend right here to splits_data before making the data frame)
    # Append ne_info to splits_data
    # for entry in splits_data:
    #     file = entry[0]
    #     sent_i = entry[1]
    #     if file in ne_info and sent_i in ne_info[file]:
    #         entry.append(ne_info[file][sent_i])
    
    # sending to get the splits WITH ne info
    print("SENDING TO GET THE SPLITS")
    splits_data = align_graphs_on_AMR_splits(all_sentences,ne_info, amr_sents,amr_graphs,umr_graphs,amr_roles, amr_roles_in_tail, umr_t2r) # changing this data structure to get the animacy info 

    # Convert splits_data to a DataFrame
    splits_data_df = pd.DataFrame(splits_data)

    columns = ["file", "sent_i","sent", "ne_info","amr_prints", "amr_graph","amr_head_name", "amr_tail_name", "amr_role","umr_head_name","umr_tail_name", "umr_role", "amr_head_id", "umr_head_id", "amr_tail_id", "umr_tail_id"] # will need to move ne_info to the end I think

    splits_data_df.columns= columns
    splits_data_df.to_csv("input_data/augment_fake_data.csv")
    return splits_data_df




        
def read_test_data():
    #reads in the raw training data, returns a df consisting of the parsed and aligned graphs
    # THIS FUNCTION IS MOSTLY COPIED OVER FROM FINALY_PROJECT.IPYNB
    # put all files in dicts
    umr_files = {}
    amr_files = {}

    umr_path = os.getcwd() + '/raw_data/UMR-data-english'
    amr_path = os.getcwd() + '/raw_data/AMR-data-english'

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
    all_sentences = {}
    for f in umr_files:
        umr_sents[f] = re.findall(r'(?<=sentence level graph:\n)\([^#]*(?=\n\n#)', umr_files[f])

    amr_sents = {}
    for f in amr_files:
        amr_sents[f] = re.findall(r'(?<=[\n])\([^#]*(?=\n|)', amr_files[f])
        sentences = re.findall(r'(?<=# ::snt\s).+?(?=\n)',amr_files[f]) #first look
        if not sentences:
            sentences = re.findall(r'(?<=:: snt)[^\n:]*(?=\n)',amr_files[f])#second look
            sentences = [re.sub(r'^\d+\s*', '', element) for element in sentences]
        all_sentences[f] = sentences


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

    ne_info = {}
    for f in all_sentences:
        ne_info[f] = parse_animacy_runner(all_sentences[f])

    splits_data = align_graphs_on_AMR_splits(all_sentences,ne_info, amr_graphs,umr_graphs,amr_roles, amr_roles_in_tail, umr_t2r)
    splits_data_df = pd.DataFrame(splits_data)

    # fill in the UMR columns with dummy variable of nothing should be fine since it is just predicting, run it through 


    columns = ["file", "sent_i","sent","ne_info", "amr_graph","amr_head_name", "amr_tail_name", "amr_role","umr_head_name","umr_tail_name", "umr_role", "amr_head_id", "umr_head_id", "amr_tail_id", "umr_tail_id"]

    splits_data_df.columns= columns
    splits_data_df.to_csv("input_data/test_data.csv")
    return splits_data_df

# make a new function that works for data augmention adding it from the FinalProject.ipynb


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
        print(text)

        # Tokenize input text and get BERT embeddings
        inputs = tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = bert_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)  # Using mean pooling for simplicity
        embeddings.append(embedding)
    b = torch.Tensor(len(data), N_BERT,D_BERT)

    return torch.cat(embeddings, out = b)

def create_mapping():
    amr_roles= {":mod",
            ":cause",
            ":part", 
            ":consist-of",
            ":source",
            ":destination",
            ":condition",
            ":ARG1-of"}

    umr_roles = {":mod", #mod with a space to avoid modal
                    ":other-role",
                    ":cause",
                    ":Cause-of",
                    ":reason",
                    ":part",
                    ":group",
                    ":material",
                    ":source",
                    ":start",
                    ":goal",
                    ":recipient",
                    ":condition",
                    ":Material-of"}

    #crete role relationship dict

    amr2umr_splits = dict.fromkeys(amr_roles,0)
    amr2umr_splits[":mod"] = [":mod",":other-role"]
    amr2umr_splits[":cause"] = [":cause",":reason"]
    amr2umr_splits[":part"] = [":part"]
    amr2umr_splits[":cause"] = [":cause",":reason"]
    amr2umr_splits[":source"] = [":material",":source",":start"]
    amr2umr_splits[":consist-of"] = [":part",":group",":material",":Material-of"]
    amr2umr_splits[":destination"] = [":goal",":recipient"]
    amr2umr_splits[":condition"] = [":condition"]
    amr2umr_splits[":ARG1-of"] = [":cause",":Cause-of"] #manipulative right now, this doesn't fully reflect split roles

    swap_amr_int_dict = create_combined_dict(amr_roles)
    swap_umr_int_dict = create_combined_dict(umr_roles)
    return convert_mapping_2_ints(amr2umr_splits,swap_amr_int_dict,swap_umr_int_dict),swap_amr_int_dict,swap_umr_int_dict

def convert_mapping_2_ints(mapping, amr, umr):
    new_mapping = {}
    for i in mapping.keys():
        values = mapping[i]
        new_mapping[amr[i]] = [umr[j] for j in values]
    return new_mapping


def create_combined_dict(input_set):
    combined_dict = {}
    for index, element in enumerate(input_set):
        combined_dict[index] = element
        combined_dict[element] = index
    return combined_dict



def preprocess_data(split, reload_graphs, reload_rules):
    #Function that will read in files from the input/test_data_splits or training data_split 
    #Return a dataframe X where X contains sentence info, graph info, animacy info, amr and umr_role, rule info
    #Args: 
        #split: string either "train" or "test"
        #reload_rules: bool, True to reprocess raw->input, creates graphs, calculate rule distriutions;  False to just load in data
            #Need True when rules have changed, or Graphs need to be accessible for doing some calcs

    rules_file = "input_data/rules_" +split+".csv"

    #load in or regenerate the files 
    if split == "train":
        if reload_graphs == True:
            X= read_training_data()
        else:
            X = pd.read_csv("input_data/train_data.csv")
            X['ne_info'] = X['ne_info'].apply(ast.literal_eval) #ne_info will need to be a literal
    elif split == "test":
        if reload_graphs == True:
            X = read_test_data()
        else:
            X = pd.read_csv("input_data/test_data.csv")
            X['ne_info'] = X['ne_info'].apply(ast.literal_eval) #ne_info will need to be a literal
    elif split == "augment":
        if reload_graphs == True:
            X = read_augment_fake_parallel_data()
        else:
            X = pd.read_csv("input_data/augment_fake_data.csv")
            X['ne_info'] = X['ne_info'].apply(ast.literal_eval) #ne_info will need to be a literal
    else:
        print("arg 1 must be 'test','train', or 'augment'")
        return
    

    if reload_rules == True:
        rules = detect_split_role(X)
        # Create a DataFrame
        rules_df = pd.DataFrame(rules, columns=['y_guess', 'y_guess_dist'])  
        rules_df.to_csv(rules_file)
    else:
        rules_df = pd.read_csv(rules_file)
        rules_df['y_guess'] = rules_df['y_guess'].apply(ast.literal_eval) 
        rules_df['y_guess_dist'] = rules_df['y_guess_dist'].apply(ast.literal_eval)
    X = pd.concat([X, rules_df], axis = 1)
    
    
    #clean up data
    a2umr_map,amr_2_int, umr_2_int = create_mapping()
    rows_to_drop = [] #if alignment stop made a mistake and found roles that we aren't exploring
    X = X.dropna(subset=['umr_role']).reset_index(drop=True) #remove missing y_true
    for i in range(len(X['umr_role'])):
        if X['umr_role'][i] not in umr_2_int:
            rows_to_drop.append(i)
        if X['amr_role'][i] not in amr_2_int:
            rows_to_drop.append(i)
    X = X.drop(rows_to_drop)
    X = X.reset_index(drop=True) #reset the indices
    return X
      


#X = preprocess_data("test", False, False)
#print(create_mapping())
# read_test_data()
# read_training_data("training_data")
# read_augment_sample_data()
read_augment_fake_parallel_data()