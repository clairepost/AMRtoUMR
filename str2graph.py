import networkx as nx
import re
import matplotlib.pyplot as plt



def clean_graph(graph_str):
    #1:clean graph takes in a string, strips the white space aroung it, makes consistent indentations
    #2:changes all ': text' to start on its own line for the sake of graph making
    graph_str = graph_str.rstrip()

    indent = 4 #just pick an indentation level, will fix after everything is on a new line
    
    #1. consistent indentation- everything will be 4 spaces
    head_spaces = {}
    spacings = []
    all_lines = graph_str.splitlines()
    #start by retrieving all spaces
    for line in all_lines:
        dist = len(line) - len(line.lstrip())
        spacings.append(dist)
    
    #Create new spacings
    no_dupes = []
    [no_dupes.append(x) for x in spacings if x not in no_dupes]
    new_spacings = [sorted(no_dupes).index(x) * 4 for x in spacings]

    #switch out the spacings
    for line_i in range(len(all_lines)):
        all_lines[line_i]=new_spacings[line_i] *' ' + all_lines[line_i].lstrip()
    graph_str = '\n'.join(all_lines)
    
    #2: splits :x to be its own line
    lines= re.findall(r'.*:.*:.*',graph_str)# find lines with multiple ":"
    for line in lines:
        #actually does the replacement on new level
        line = line.split(':')
        for i in range(2,len(line)):
            num_spaces = len(line[0]) - len(line[0].lstrip())
            new_str = '\n' + ' ' * num_spaces + indent * ' ' + ":"
            graph_str = re.sub(r'(?<=[^\s]) :', new_str,graph_str,count = 1)    

    return graph_str

def draw_graph(G):
    #draws a graph, planar using attribute 'name' as the node label, and edge attribute 'label' as the edge labels
    pos = nx.planar_layout(G)
    plt.figure(1,figsize=(12,12)) 
    nx.draw(G,pos, with_labels = True,labels = nx.get_node_attributes(G, 'name'),node_size=60,font_size=8)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=nx.get_edge_attributes(G,'label'),font_size=8)
    return


def get_head_spacing(curr_spacing, head_dict):
    #get the head of current line based on spacing
    all_spacings = list(head_dict.keys())
    all_spacings.sort()
    i = all_spacings.index(curr_spacing)
    head_spacing = all_spacings[i-1]
    return head_spacing
    


def extract_node(node_str,i):
    #get node, it's id and edge relation from the line (node_str)
    # i (the line the nodestr is on) will be used as a unique identifier
    node_str = node_str.strip()
    node_list = node_str.split(' / ')
    if len(node_list) == 1:
        #no name, just a reference to an id within the graph
        #print(node_list)
        #create new id for the graph so that we don't accidently duplicate the nodes when adding
        node_list= node_list[0].split(" ")
        node = node_list[1]
        edge = node_list[0]
        node = node.strip(" ()")
        g_id = node +"_"+ str(i)


    else:
        node = node_list[-1].strip(')')

        if node_list[0][0] != ":":
            edge = None
            g_id = node_list[0].strip(" ()") 
            g_id = g_id +"_" +str(i)
        else:
            [edge,g_id] = node_list[0].split(" (")
            g_id = g_id +"_" +str(i)
    return node, g_id,edge

def create_graph(graph_str):
    head_dict = {}
    G = nx.DiGraph()

    graph_str = clean_graph(graph_str)
    for i in range(len(graph_str.splitlines())):
        #extact node and edge, get spacing to know what to attach it to 
        curr_line = graph_str.splitlines()[i]
        curr_node, g_id, edge = extract_node(curr_line,i) 
        curr_spacing = len(curr_line) - len(curr_line.lstrip())
        head_dict[curr_spacing] = g_id #update thespacing level to point to this node
        if i == 0:
            #add top  node to the graph
            G.add_node(g_id,name = curr_node, head_id = None)
        else:
            #get head_node spacing
            head_spacing = get_head_spacing(curr_spacing, head_dict)
            head_node_id= head_dict[head_spacing]
        
            #attach curr_node to head_node
            G.add_node(g_id,name=curr_node, head= head_node_id)
            G.add_edge(head_node_id,g_id, label = edge)#add whatever is last in the heap
            
    return G


            

        