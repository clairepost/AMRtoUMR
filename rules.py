import re

# Rules for split role determination.

def detect_split_role(X_tuples): 

    Y = []

    for x_tuple in X_tuples:

        # getting info from X_tuple
        sentence = x_tuple[0]
        named_entity = x_tuple[1]
        amr_graph = x_tuple[2]
        head = x_tuple[3] 
        role = x_tuple[4]
        tail = x_tuple[5]
        # check for cause-01
        if tail == "cause-01":
            # update the tail
            tail, role, head = fixCause(sentence, amr_graph, tail, role, head)
        #get animacy
        animacy_info = ne_animacy(named_entity, tail)


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
            parent_node = get_parent_node(role, amr_graph)  # Implement a function to get the parent node
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

    # Check if named_entity is empty or contains ANY animate roles
    for ne_entry in named_entity:
        for ne_type, ne_value in ne_entry.items():
            if ne_type in ["PER", "B_human", "B_animal"]:
                return "animate"

    # Default to Inanimate if no match is found
    return "inanimate"


def fixCause(sentence, amr_graph, tail, role, head):
    # this function will fix the graph when cause-01 appears
    child_nodes = [child for _, _, child in amr_graph if _ == head]

    # Check if any child node is a verb ("-01" or other "-##" patterns)
    verb_child = next((child for child in child_nodes if re.search(r'-\d+$', child)), None)

    # If there is a verb child, recursively call fixCause on that child
    if verb_child:
        return fixCause(sentence, amr_graph, verb_child, role, head)
    else:
        # If no verb child is found, return the first child as the new tail
        new_tail = child_nodes[0] if child_nodes else tail
        return new_tail, role, head
    return tail, role


def get_parent_node(amr_graph,head):
    """
    Helper function to get the parent node of a given role instance.

    Returns:
    - parent_node (str): The parent node of the given role instance.
    """

    if ":theme" in amr_graph:
        return ":theme"
    
    # Find the role to the head of the current role (the current role's parent role)
    for edge in amr_graph:
        if edge[2] == head:  # Check if the third element (head) of the edge matches the given head
            return edge[1]  # Return the second element (parent node) of the edge
    return None  # Return None if no parent node is found
    

