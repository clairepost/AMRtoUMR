from nltk.stem import PorterStemmer

def align_graphs_on_AMR_splits(sentences, amr_graphs,umr_graphs,amr_roles,amr_roles_in_tail,umr_t2r):
    cant_find_heads_count = 0
    cant_find_tails_count = 0
    umr_head_tail_no_role = 0
    total_count=0
    porter = PorterStemmer()
    splits_data = []
    for file in amr_graphs.keys():
        for sent_i in range(len(amr_graphs[file])):
            amr_graph = amr_graphs[file][sent_i]
            umr_graph = umr_graphs[file][sent_i]

            sent = sentences[sent_i]
            
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
                    tail_matcher = porter.stem(tail_ans[1].split('-')[0])
                    head_matcher = porter.stem(amr_head_name.split('-')[0])

                    if r in amr_roles_in_tail:
                       if amr_tail_name == amr_roles_in_tail[r]:
                            
                            head_ans = [item for item in (umr_graph.nodes(data="name")) if porter.stem(item[1].split('-')[0]) == head_matcher or item[1]== amr_head_name]
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
                                entry = [file, sent_i, sent, amr_graph, amr_head_name, amr_tail_name, amr_role, umr_head_name, umr_tail_name, umr_role, amr_head_id, umr_head_id, amr_tail_id, umr_tail_id]
                                splits_data.append(entry)
                    elif (edge[2]==r and r not in amr_roles_in_tail): # found an amr split role in the graph we're looking for, now let's align it to the umr graph
                        total_count +=1
                        #continue on
                       
                        #get matching umr info, check if node when split is equal to the matcher or just the basic version
                        head_ans = [item for item in (umr_graph.nodes(data="name")) if porter.stem(item[1].split('-')[0]) == head_matcher or item[1]== amr_head_name]
                        if head_ans: #if head is found
                            umr_head_id = head_ans[0][0]
                            umr_head_name= head_ans[0][1]
                            
                            tail_ans = [item for item in (umr_graph.nodes(data="name")) if porter.stem(item[1].split('-')[0]) == tail_matcher]
                            if tail_ans:#if tail was found
                                umr_tail_id = tail_ans[0][0]
                                umr_tail_name = tail_ans[0][1]
                                umr_role = umr_graph.get_edge_data(umr_head_id,umr_tail_id)
                                if umr_role: #if umr role was found
                                    umr_role = umr_role["label"] #will return none if no edge
                                else:
                                    umr_head_tail_no_role+=1

                                #create entry and add to data
                                entry = [file, sent_i, sent, amr_graph, amr_head_name, amr_tail_name, amr_role, umr_head_name, umr_tail_name, umr_role, amr_head_id, umr_head_id, amr_tail_id, umr_tail_id]
                                splits_data.append(entry)
                            else:
                                #couldn't find matching tail in umr graph
                                cant_find_tails_count+=1
                                print("missing tail edge:", edge, "in File",file, "sentence", sent_i )
                        else:
                            #coudn't find matching head in umr graph
                            cant_find_heads_count+=1


                            print("missing head edge:", (amr_head_name,amr_tail_name,amr_role), "in File",file, "sentence", sent_i )

    print("unable to find", cant_find_heads_count," matching UMR heads")
    print("unable to find", cant_find_tails_count,"matching UMR tails")
    print("unable to find", umr_head_tail_no_role,"link betwen head-tail")
    print("total amr split roles examined: ", total_count)
    return splits_data
                        