from nltk.stem import WordNetLemmatizer
import ast

def align_graphs_on_AMR_splits(sentence_dict, ne_info, amr_print, amr_graphs,umr_graphs,amr_roles,amr_roles_in_tail,umr_t2r):


    cant_find_heads_count = 0
    cant_find_tails_count = 0
    umr_head_tail_no_role = 0
    total_count=0
    lemmatizer = WordNetLemmatizer()
    splits_data = []
    for file in amr_graphs.keys():
        sentences = sentence_dict[file]
        for sent_i in range(len(amr_graphs[file])):
            amr_graph = amr_graphs[file][sent_i]
            umr_graph = umr_graphs[file][sent_i]
            amr_prints = amr_print[file][sent_i] # added amr printed graph

            # # If umr_graphs is None or empty, set umr_graph to None
            # if umr_graphs is None or not umr_graphs.get(file):
            #     umr_graph = None
            # else:
            #     umr_graph = umr_graphs[file][sent_i]

            sent = sentences[sent_i]
            ne = ne_info[file][sent_i]
            
            if type(ne) == str:
                ne = ast.literal_eval(ne)[0]

            #get edge where edge in AMR roles
            amr_all_edges = amr_graph.edges(data='label')
            for r in amr_roles:
                for edge in amr_all_edges:
                   
                    amr_role = edge[2]
                    
                    #get amr head and tail node and ids 
                    head_ans = [item for item in (amr_graph.nodes(data="name")) if item[0] == edge[0]][0]
                    amr_head_id = head_ans[0]
                    amr_head_name = head_ans[1]

                    tail_ans = [item for item in (amr_graph.nodes(data="name")) if item[0] == edge[1]][0]
                    amr_tail_id = tail_ans[0]
                    amr_tail_name =  tail_ans[1]

                    #split on the '-' so that if there's an updated roleset it won't matter, also do a stemmer in case there's slight variation
                    tail_matcher = lemmatizer.lemmatize(tail_ans[1].split('-')[0])
                    head_matcher = lemmatizer.lemmatize(amr_head_name.split('-')[0])

                    if r in amr_roles_in_tail:
                       if amr_tail_name == amr_roles_in_tail[r]:
                            
                            head_ans = [item for item in (umr_graph.nodes(data="name")) if lemmatizer.lemmatize(item[1].split('-')[0]) == head_matcher or item[1]== amr_head_name]
                            # get the relation and then we will get the tail
                            r_matcher =  umr_t2r[amr_tail_name]
                            #r_ans = [item for item in list(umr_graph.edges(data='label'))[2] if porter.stem(item.split('-')[0]) in r_matcher or item in r_matcher]
                            r_ans = []
                            for item in list(umr_graph.edges(data='label')):
                                if item[2] is not None and item[2] in r_matcher:
                                    r_ans=(item)
                                    umr_role = r_ans[2]
                            if r_ans:
                                total_count +=1
                                umr_head_name = umr_graph.nodes[r_ans[0]]['name']
                                umr_tail_name = umr_graph.nodes[r_ans[1]]['name']
                                umr_tail_id = r_ans[0]
                                umr_head_id = r_ans[1]
                                entry = [file, sent_i, sent, ne,amr_prints, amr_graph, amr_head_name, amr_tail_name, amr_role, umr_head_name, umr_tail_name, umr_role, amr_head_id, umr_head_id, amr_tail_id, umr_tail_id]
                                splits_data.append(entry)
                    elif (edge[2]==r and r not in amr_roles_in_tail): # found an amr split role in the graph we're looking for, now let's align it to the umr graph
                        total_count +=1
                        #continue on
                       
                        #get matching umr info, check if node when split is equal to the matcher or just the basic version
                        head_ans = [item for item in (umr_graph.nodes(data="name")) if lemmatizer.lemmatize(item[1].split('-')[0]) == head_matcher or item[1]== amr_head_name]
                        if head_ans: #if head is found
                            umr_head_id = head_ans[0][0]
                            umr_head_name= head_ans[0][1]
                            
                            tail_ans = [item for item in (umr_graph.nodes(data="name")) if lemmatizer.lemmatize(item[1].split('-')[0]) == tail_matcher]
                            if tail_ans:#if tail was found
                                umr_tail_id = tail_ans[0][0]
                                umr_tail_name = tail_ans[0][1]
                                umr_role = umr_graph.get_edge_data(umr_head_id,umr_tail_id)
                                if umr_role: #if umr role was found
                                    umr_role = umr_role["label"] #will return none if no edge
                                else:
                                    umr_head_tail_no_role+=1

                                #create entry and add to data
                                entry = [file, sent_i, sent, ne,amr_prints, amr_graph, amr_head_name, amr_tail_name, amr_role, umr_head_name, umr_tail_name, umr_role, amr_head_id, umr_head_id, amr_tail_id, umr_tail_id]
                                splits_data.append(entry)
                            else:
                                #couldn't find matching tail in umr graph
                                cant_find_tails_count+=1
                                print("missing tail edge:", edge, "in File",file, "sentence", sent_i )
                        else:
                            #coudn't find matching head in umr graph
                            cant_find_heads_count+=1


                            print("missing head edge:", (amr_head_name,amr_tail_name,amr_role), "in File",file, "sentence", sent_i )

    print("unable to find", cant_find_tails_count,"matching UMR tails")
    print("unable to find", umr_head_tail_no_role,"link betwen head-tail")
    print("total amr split roles examined: ", total_count)
    return splits_data

def align_graphs_no_animacy(sentence_dict, ne_info, amr_print, amr_graphs,umr_graphs,amr_roles,amr_roles_in_tail,umr_t2r):

    cant_find_heads_count = 0
    cant_find_tails_count = 0
    umr_head_tail_no_role = 0
    total_count=0
    lemmatizer = WordNetLemmatizer()

    all_amr_roles = amr_roles.copy() 
    all_amr_roles.update(amr_roles_in_tail)

    splits_data = []
    for file in amr_graphs.keys():
        sentences = sentence_dict[file]
        for sent_i in range(len(amr_graphs[file])):
            amr_graph = amr_graphs[file][sent_i]
            umr_graph = umr_graphs[file][sent_i]
            amr_prints = amr_print[file][sent_i] # added amr printed graph

            # # If umr_graphs is None or empty, set umr_graph to None
            # if umr_graphs is None or not umr_graphs.get(file):
            #     umr_graph = None
            # else:
            #     umr_graph = umr_graphs[file][sent_i]

            sent = sentences[sent_i]
            ne = ne_info[file][sent_i]
            
            if type(ne) == str:
                ne = ast.literal_eval(ne)[0]

            #get edge where edge in AMR roles
            amr_all_edges = amr_graph.edges(data='label')
            for r in amr_roles:
            # for r in all_amr_roles:
                for edge in amr_all_edges:
                   
                    amr_role = edge[2]
                    
                    #get amr head and tail node and ids 
                    head_ans = [item for item in (amr_graph.nodes(data="name")) if item[0] == edge[0]][0]
                    amr_head_id = head_ans[0]
                    amr_head_name = head_ans[1]
                    

                    tail_ans = [item for item in (amr_graph.nodes(data="name")) if item[0] == edge[1]][0]
                    amr_tail_id = tail_ans[0]
                    amr_tail_name =  tail_ans[1]

                    #split on the '-' so that if there's an updated roleset it won't matter, also do a stemmer in case there's slight variation
                    tail_matcher = lemmatizer.lemmatize(tail_ans[1].split('-')[0])
                    head_matcher = lemmatizer.lemmatize(amr_head_name.split('-')[0])

                    if r in amr_roles_in_tail:
                       
                       if amr_tail_name == amr_roles_in_tail[r]:
                            print("\nIN THE AMR ROLES IN TAIL\n", "r: ", r, "roles: ", amr_roles_in_tail)
                            print("\nsecond edge: ", amr_role)
                            print("amr_head_name: ", amr_head_name)
                            print("amr_tail_name: ", amr_tail_name)
                            print("checking if the tail name is in the amr_roles_in_tail == TRUE")
                            head_ans = [item for item in (umr_graph.nodes(data="name")) if lemmatizer.lemmatize(item[1].split('-')[0]) == head_matcher or item[1]== amr_head_name]
                            # get the relation and then we will get the tail
                            r_matcher =  umr_t2r[amr_tail_name]
                            # flag = False
                            r_ans = []
                            for item in list(umr_graph.edges(data='label')):
                                tail_check = umr_graph.nodes[item[1]]['name']
                                if item[2] is not None and item[2] in r_matcher:
                                    print("found ", item[2], " in tail checker")
                                    r_ans=(item)
                                    umr_role = r_ans[2]
                                    print("umr role: ", umr_role)
                                    
                                elif tail_check in r_matcher:
                                    print("found ", tail_check, " in tail checker")
                                    r_ans= (item)
                                    umr_role = tail_check
                                
                                    
                            # if flag == True
                            if r_ans:
                                print("r_ans: ", r_ans)
                                print("umr_role: ", umr_role)
                                print("FileID: ", file)
                                print(sent)
                                print(amr_prints)
                                total_count +=1
                                umr_head_name = umr_graph.nodes[r_ans[0]]['name']
                                umr_tail_name = umr_graph.nodes[r_ans[1]]['name']
                                umr_tail_id = r_ans[0]
                                umr_head_id = r_ans[1]
                                entry = [file, sent_i, sent, ne,amr_prints, amr_graph, amr_head_name, amr_tail_name, amr_role, umr_head_name, umr_tail_name, umr_role, amr_head_id, umr_head_id, amr_tail_id, umr_tail_id]
                                splits_data.append(entry)
                                print("\nFOUND A Cause: ", entry,"\n")
                            # else:
                            #     umr_head_tail_no_role+=1
                            #     umr_head_name = "None"
                            #     umr_tail_name = "None"
                            #     umr_tail_id = "None"
                            #     umr_head_id = "None"
                            #     #create entry and add to data
                            #     entry = [file, sent_i, sent, ne,amr_prints, amr_graph, amr_head_name, amr_tail_name, amr_role, umr_head_name, umr_tail_name, umr_role, amr_head_id, umr_head_id, amr_tail_id, umr_tail_id]
                            #     splits_data.append(entry)
                    elif (edge[2]==r and r not in amr_roles_in_tail): # found an amr split role in the graph we're looking for, now let's align it to the umr graph
                        total_count +=1
                        #continue on
                       
                        #get matching umr info, check if node when split is equal to the matcher or just the basic version
                        head_ans = [item for item in (umr_graph.nodes(data="name")) if lemmatizer.lemmatize(item[1].split('-')[0]) == head_matcher or item[1]== amr_head_name]
                        if head_ans: #if head is found
                            umr_head_id = head_ans[0][0]
                            umr_head_name= head_ans[0][1]
                            
                            tail_ans = [item for item in (umr_graph.nodes(data="name")) if lemmatizer.lemmatize(item[1].split('-')[0]) == tail_matcher]
                            if tail_ans:#if tail was found
                                umr_tail_id = tail_ans[0][0]
                                umr_tail_name = tail_ans[0][1]
                                umr_role = umr_graph.get_edge_data(umr_head_id,umr_tail_id)
                                if umr_role: #if umr role was found
                                    umr_role = umr_role["label"] #will return none if no edge
                                else:
                                    umr_head_tail_no_role+=1

                                #create entry and add to data
                                entry = [file, sent_i, sent, ne,amr_prints, amr_graph, amr_head_name, amr_tail_name, amr_role, umr_head_name, umr_tail_name, umr_role, amr_head_id, umr_head_id, amr_tail_id, umr_tail_id]
                                splits_data.append(entry)
                            else:
                                #couldn't find matching tail in umr graph
                                cant_find_tails_count+=1
                                print("missing tail edge:", edge, "in File",file, "sentence", sent_i )
                        else:
                            #coudn't find matching head in umr graph
                            cant_find_heads_count+=1


                            print("missing head edge:", (amr_head_name,amr_tail_name,amr_role), "in File",file, "sentence", sent_i )

    print("unable to find", cant_find_tails_count,"matching UMR tails")
    print("unable to find", umr_head_tail_no_role,"link betwen head-tail")
    print("total amr split roles examined: ", total_count)
    return splits_data


                        