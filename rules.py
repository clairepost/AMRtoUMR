import re
import networkx as nx
from animacyParser import parse_animacy_runner

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
            tail, role = fixCause(amr_graph, tail, role)
        #get animacy
        animacy_info = ne_animacy(named_entity, tail, amr_graph, sentence)
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
            Y.append(([":mod", ":other-role"], [0.99, 0.01]))

        # Rule 8: condition (concessive condition does not really occur in English)
        elif role == ":condition":
            Y.append(([":condition"], [1.0]))

        # Default: If no rule is matched, return None
        else:
            Y.append((["None"], [1.0]))
        print("Y value: ",Y[-1])


    print("Y:\n",Y)
    return Y



def ne_animacy(named_entity, tail, amr_graph, sentence):

    # Check if named_entity matches tail
    animacy = animacy_classification(named_entity, tail)
    if animacy != "none":
        return animacy

    # just check the animacy of the tail now if we could not find anything
    print("sending tail to be parsed:", tail)
    new_named_entity = parse_animacy_runner([tail]) # needs to send sentence and tail
    print("NEW NAMED ENTITY: ", new_named_entity)

    # Extract the first dictionary from the list, if any
    if new_named_entity and isinstance(new_named_entity[0], dict):
        new_named_entity = new_named_entity[0]
        print("change to new_named_entity:", new_named_entity)
    else:
        new_named_entity = {}  # Default to an empty dictionary if no valid dictionary is found
    animacy = animacy_classification(new_named_entity, tail)
    if animacy != "none":
        print("used new animacy identified from running animacy parser again")
        return animacy

    # Run through a second time on the next entity if it was a verb
    animacy = second_pass_animacy(named_entity, tail, amr_graph)
    if animacy != "none":
        return animacy

    # Default to Inanimate if no match is found
    return "inanimate"

def animacy_classification(named_entity, tail):
    print("Named entities going into animacy classification: ", named_entity)
    # Check if named_entity matches tail
    for ne_entry in named_entity:
        for ne_type, ne_value in ne_entry.items():
            if ne_value == tail:
                # Check animacy based on ne_type
                if ne_type in ["PER", "B_human", "B_animal"]:
                    print("returning animate!!")
                    return "animate"
                elif ne_type in ["ORG", "LOC", "MISC"]:
                    print("returning inanimate!!")
                    return "inanimate"
    return "none"

def second_pass_animacy(named_entity, tail, amr_graph):
    # checking the next node if it is -## assuming it is child of verb
    print("\nin second pass animacy\n")
    if re.search(r'-\d+$', tail):
        # get the label name for find replacement
        role_id = get_label_name(amr_graph, tail)
        # check for the child of this node that is not a verb
        second_check_node = find_replacement_node(amr_graph, tail, role_id)
        print("second check node: ", second_check_node)
        # use the child node in the new animacy classification
        second_animacy = animacy_classification(named_entity, second_check_node)
        print("second animacy term: ", second_animacy)
        # if there was no animacy run the parse animacy runner again on the child node
        if second_animacy == "none":
            new_named_entity = parse_animacy_runner([second_check_node])
            # Extract the first dictionary from the list, if any
            if new_named_entity and isinstance(new_named_entity[0], dict):
                new_named_entity = new_named_entity[0]
                print("change to new_named_entity (second pass):", new_named_entity)
            else:
                new_named_entity = {}  # Default to an empty dictionary if no valid dictionary is found
            print("new_named_entity: ",new_named_entity)
            # then return the new animacy classification
            return animacy_classification(new_named_entity, second_check_node)
        else:
            return second_animacy
   
    return "none"


def fixCause(amr_graph, tail, role):
    # re.search(r'-\d+$', successor)

    print("\nSearching new role\n")
    print("amr graph nodes: ", amr_graph.nodes(data="name"))
    print("amr graph edges: ", amr_graph.edges(data=True))
    # get the role id for cause-01
    role_id = get_label_name(amr_graph, tail)
    print("role_id: ", role_id)
    replacement_node = find_replacement_node(amr_graph, tail, role_id)
    print("\nFOUND NEW CAUSE ROLE:\n", replacement_node)

    return replacement_node, ":cause"

def find_replacement_node(amr_graph, tail, role_id):
    # Find the child node of the current node with the specified role and head
    nodes_list = list(amr_graph.nodes(data="name"))
    node_found1 = False
    for node in nodes_list:
        if node_found1:
            child_node_id = node[1]
            print("matches and we've got child node:", node[1])
            # Check if the child node is a verb ("-01" or other "-##" pattern)
            if re.search(r'-\d+$', child_node_id):
                continue
            else:
                return child_node_id #otherwise return the new tail
        elif node[0] == role_id:
            node_found1 = True
    
    # If no suitable replacement is found, return the next item in the nodes list
    node_found = False
    for node in nodes_list:
        if node_found:
            print("just returning next node:", node[1])
            return node[1]
        if node[0] == role_id:
            node_found = True
    
    # otherwise just return the tail again
    return tail


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


