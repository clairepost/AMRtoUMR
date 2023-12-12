import re
import networkx as nx

# Rules for split role determination.

def detect_split_role(X_tuples): 

    Y = []
    print("X_tuples 0:")
    print(X_tuples)
    count = 0

    for _, x_tuple in X_tuples.iterrows():  # Using iterrows to iterate over DataFrame rows
        count+=1
        # getting info from X_tuple
        print("\nSentence ", count, ":\n")
        print("sent: \n", x_tuple["sent"])
        print("ne_info: \n", x_tuple["ne_info"])
        print("amr_graph: \n", x_tuple["amr_graph"])
        print("amr_head_name: \n", x_tuple["amr_head_name"])
        print("amr_role: \n", x_tuple["amr_role"])
        print("amr_tail_name: \n", x_tuple["amr_tail_name"])

        # getting each piece of info from X_tuple
        sentence = x_tuple["sent"]
        named_entity = x_tuple["ne_info"]
        amr_graph = x_tuple["amr_graph"]
        head = x_tuple["amr_head_name"]
        role = x_tuple["amr_role"]
        tail = x_tuple["amr_tail_name"]
        # check for cause-01
        if tail == "cause-01":
            # update the tail
            tail, role, head = fixCause(sentence, amr_graph, tail, role, head)
        #get animacy
        animacy_info = ne_animacy(named_entity, tail)
        print("animacy: \n", animacy_info)


        # Rule 1: :destination instance
        if role == ":destination":
            if animacy_info == "inanimate":
                Y.append(([":goal"], [1.0]))
            elif animacy_info == "animate":
                Y.append(([":goal", ":recipient"], [0.1, 0.9]))

        # Rule 2: :cause instance
        elif role == ":cause":
            if animacy_info == "inanimate":
                Y.append(([":cause", ":reason"], [0.9, 0.1]))
            elif animacy_info == "animate":
                Y.append(([":reason"], [1.0]))

        # Rule 3: Source -> (material, source, start) # Rule 4: Find parent node of source and check if it has a theme
        elif role == ":source":
            parent_node = get_parent_node(amr_graph, head, tail)  # Implement a function to get the parent node
            if animacy_info == "animate":
                Y.append(([":source"], [1.0]))
            elif parent_node == ":theme":
                Y.append(([":source"], [1.0]))
            elif "LOC" in named_entity:
                Y.append(([":source", ":start"], [0.6, 0.4]))
            elif animacy_info == "inanimate":
                Y.append(([":material", ":source", ":start"], [0.1, 0.6, 0.3]))

        # Rule 5: Find :consist-of instance
        elif role == ":consist-of":
            if animacy_info == "inanimate":
                Y.append(([":group", ":part", ":material"], [0.1, 0.1, 0.8]))
            elif animacy_info == "animate":
                Y.append(([":group"], [1.0]))

        # Rule 6: part is always part
        elif role == ":part":
            Y.append(([":part"], [1.0]))

        # Rule 7: mod
        elif role == ":mod":
            Y.append(([":mod", ":other-role"], [0.95, 0.05]))

        # Rule 8: condition (concessive condition does not really occur in English)
        elif role == ":condition":
            Y.append(([":condition"], [1.0]))

        # Default: If no rule is matched, return None
        else:
            Y.append((["None"], [1.0]))
        print("Y value: ",Y[-1])


    print("Y:\n",Y)
    return Y



def ne_animacy(named_entity, tail):

    # Check if named_entity matches tail
    for ne_entry in named_entity:
        for ne_type, ne_value in ne_entry.items():
            if ne_value == tail:
                # Check animacy based on ne_type
                if ne_type in ["PER", "B_human", "B_animal"]:
                    return "animate"
                elif ne_type in ["ORG", "LOC", "MISC"]:
                    return "inanimate"

    # # Check if named_entity is empty or contains ANY animate roles
    # for ne_entry in named_entity:
    #     for ne_type, ne_value in ne_entry.items():
    #         if ne_type in ["PER", "B_human", "B_animal"]:
    #             return "animate"

    # Default to Inanimate if no match is found
    return "inanimate"


def fixCause(sentence, amr_graph, tail, role, head):

    # find the child node of tail in the amr_graph
    # if the child node has any "-01" or any other "-##" dash plus numbers at the end then it is a verb and we need to recursively go down the children of the node until we find a tail without a node in the tuple. 
    # If we go through the whole graph and none can be found then we just return the first child 
    role = ":cause"
    return tail, role, head


def get_parent_node(amr_graph, head, tail):

    label_name_head = get_label_name(amr_graph, head)
    label_name_tail = get_label_name(amr_graph, tail)

    # Iterate over edges and find the parent role based on label names
    for edge in amr_graph.edges(data=True):

        if (label_name_head and label_name_tail) in edge:
            print("Edge: ",edge[2])
            label = edge[2]
            return label['label']
    
    if ":theme" in amr_graph.edges(data=True):
        return ":theme"

    return None

def get_label_name(amr_graph, node_id):
    # Find the label name associated with a node ID
    for node in amr_graph.nodes(data="name"):
        if node_id in node:
            return node[0]
    return None


