# Rules for split role determination.

def detect_split_role(role_instance, animacy_information, named_entity_type):
    """
    Detect split role based on the provided role instance, animacy information, and named entity type.

    Parameters:
    - role_instance (str): The instance attached to the role.
    - animacy_information (dict): Animacy information parsed from the sentence.
    - named_entity_type (str): The named entity type associated with the role.

    Returns:
    - split_role (str): The detected split role or a suggestion of possible roles.
    """

    # Rule 1: Find :destination instance
    if role_instance == ":destination":
        if animacy_information == "inanimate":
            return ":goal"
        elif animacy_information == "animate":
            return [":goal", ":recipient"]

    # Rule 2: Find :cause instance
    if role_instance == ":cause":
        if animacy_information == "inanimate":
            return [":cause", ":reason"]
        elif animacy_information == "animate":
            return ":reason"

    # Rule 3: Source -> (material, source, start)
    if role_instance in [":material", ":source", ":start"]:
        if animacy_information == "inanimate":
            return [":material", ":source", ":start"]
        elif animacy_information == "animate":
            return ":source" if named_entity_type != "LOC" else [":start", ":source"]

    # Rule 4: Find parent node of source and check if it has a theme
    if role_instance == ":source":
        parent_node = get_parent_node(role_instance)  # Implement a function to get the parent node
        if parent_node == ":theme":
            return ":source"
    
    # Rule 5: Find :cause instance
    if role_instance == ":consist-of":
        if animacy_information == "inanimate":
            return [":group", ":part-of"]
        elif animacy_information == "animate":
            return ":group"

    # Default: If no rule is matched, return None or provide a default suggestion
    return None


def get_parent_node(role_instance, umr_graph):
    """
    Helper function to get the parent node of a given role instance.

    Parameters:
    - role_instance (str): The role instance for which to find the parent node.
    - umr_graph: probably need some way to access the previous tuple that this child tuple is connencted to.

    Returns:
    - parent_node (str): The parent node of the given role instance.
    """
    # Find the incoming edges to the role_instance node and get the source nodes

    incoming_edges = umr_graph.in_edges(role_instance)
    if incoming_edges:
        parent_node = incoming_edges[0][0]
        return parent_node
    else:
        return None
