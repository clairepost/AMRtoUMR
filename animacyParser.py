# Use a pipeline as a high-level helper
import string
import re
from transformers import pipeline


pronouns = ['i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them']


# just check if these are in the tail
# maybe need to look for wiki, just check the tail node
animate_wiki = ['person', 'family', 'animal', 'ethnic-group', 'regional-group', 'religious-group', 'political-movement', 'organization', 'company', 'government-organization', 'military', 'criminal-organization', 'political-party', 'market-sector', 'school', 'university', 'research-institute', 'team', 'league']

inanimate_wiki = ['language', 'nationality', 'location', 'city', 'city-district', 'county', 'state', 'province', 'territory', 'country', 'local-region', 'country-region', 'world-region', 'continent', 'ocean', 'sea', 'lake', 'river', 'gulf', 'bay', 'strait', 'canal; peninsula', 'mountain', 'volcano', 'valley', 'canyon', 'island', 'desert', 'forest moon', 'planet', 'star', 'constellation', 'facility', 'airport', 'station', 'port', 'tunnel', 'bridge', 'road', 'railway-line', 'canal', 'building', 'theater', 'museum', 'palace', 'hotel', 'worship-place', 'market', 'sports-facility', 'park', 'zoo', 'amusement-park', 'event', 'incident', 'natural-disaster', 'earthquake', 'war', 'conference', 'game', 'festival', 'product', 'vehicle', 'ship', 'aircraft', 'aircraft-type', 'spaceship', 'car-make', 'work-of-art', 'picture', 'music', 'show', 'broadcast-program', 'publication', 'book', 'newspaper', 'magazine', 'journal', 'natural-object', 'award', 'law', 'court-decision', 'treaty', 'music-key', 'musical-note', 'food-dish', 'writing-script', 'variable', 'program', 'thing']


def parse_by_pipe(sentences, pipe, keep_list = None, raw = False):
    info = []
    for i in sentences:
        parse = pipe(i)
        if keep_list != None:
            parse_keep_only = []
            for j in parse:
                if j["entity_group"] in keep_list:
                    parse_keep_only.append(j)
            info.append(parse_keep_only)
        else:
            info.append(parse)
    
    
    #format the parse so irrelevent info is removed
    if not raw:
        results_formatted = []
        for result_i in range(len(info)):
            result_f = []
            for ne_dict in info[result_i]:
                # Start and end provide an easy way to highlight words in the original text.
                a = sentences[result_i][ne_dict["start"] : ne_dict["end"]]
                b = ne_dict['entity_group']
                result_f.append({b:a})
            results_formatted.append(result_f)
        return results_formatted
    else:
        return info
    
def parse_by_pipe_splitroles_only(sentences, amr_graphs, pipe, keep_list = None, raw = False):
    info = []
    for i, sentence in enumerate(sentences):

        # Retrieve the corresponding AMR graph for the current sentence
        amr_graph = amr_graphs[i]
        has_split_role = splitrole_check(amr_graph)

        if has_split_role:
            parse = pipe(sentence)
            if keep_list != None:
                parse_keep_only = []
                for j in parse:
                    if j["entity_group"] in keep_list:
                        parse_keep_only.append(j)
                info.append(parse_keep_only)
            else:
                info.append(parse)
        else: # so we don't call pipeline if there is no splitrole in sentence
            info.append([])
    
    #format the parse so irrelevent info is removed
    if not raw:
        results_formatted = []
        for result_i in range(len(info)):
            result_f = []
            for ne_dict in info[result_i]:
                # Start and end provide an easy way to highlight words in the original text.
                a = sentences[result_i][ne_dict["start"] : ne_dict["end"]]
                b = ne_dict['entity_group']
                result_f.append({b:a})
            results_formatted.append(result_f)
        return results_formatted
    else:
        return info


def splitrole_check(amr_graph):
    amr_roles= {
        ":cause",
        "cause-01",
        ":cause-01",
        ":part", 
        ":consist-of",
        ":source",
        ":destination",
        ":condition",
        ":ARG1-of"} # ignoring :mod
    # Check if the word matches any of the roles in amr_roles

    for word in amr_graph.split():
        if word in amr_roles:
            return True
    # If no split role is found, return False
    return False


def wiki_type(sentences,amr_graphs, animate = animate_wiki, inanimate = inanimate_wiki):
    info = []
    for amr_graph in amr_graphs:
        # amr_graph = amr_graphs[i]
        input_amr = remove_punctuation(amr_graph)
        word_list = input_amr.split(" ")
        wiki_found = []
        for i in word_list:
            if i.lower() in animate:
                wiki_found.append({"W_Animate":i})
            elif i.lower() in inanimate:
                wiki_found.append({"W_Inanimate":i})
        info.append(wiki_found)
    return info

def parse_for_pronouns(sentences,pronouns= pronouns):
    info = []
    for input_sent in sentences:
        input_sent = remove_punctuation(input_sent)
        word_list = input_sent.split(" ")
        pn_found = []
        for i in word_list:
            if i.lower() in pronouns:
                pn_found.append({"PER":i})
        info.append(pn_found)
    return info

def combine_parses(list_of_parses):
    #takes in a list of lists containing the parse info for each sentence. All interior lists have to be the same length
    num_parses = len(list_of_parses)
    for i in range(len(list_of_parses[0])):
        for j in range(1,num_parses):
            list_of_parses[0][i].extend(list_of_parses[j][i])

    combined = list_of_parses[0]
    # for i in range(len(combined)):
    #     if combined[i] == []:
    #         combined[i] = [{}]
    # Additional check to return an empty list if all items are empty lists
    if all(item == [] for item in combined):
        combined = []

    print("COMBINED: ", combined)
    
    return combined



def remove_punctuation(input_string):
    # Create a translation table for removing punctuation
    translator = str.maketrans('', '', string.punctuation)
    
    # Apply the translation to the input string
    result = input_string.translate(translator)
    
    return result




# def parse_animacy_runner(sentences, amr_print, amr_graph, tail, role):
# def parse_animacy_runner(sentences, amr_print, X_tuples, file_id):
def parse_animacy_runner(sentences, amr_print):
    ##input should be a list, either a list of sentences, a list containing words (could be one word)
    #returns list of dicts that contain combined aimacy, pronoun, and ner info

    #create pipelines
    pipe_animacy = pipeline("token-classification", model="andrewt-cam/bert-finetuned-animacy",aggregation_strategy="simple")
    pipe_ner = pipeline("token-classification", model="dslim/bert-base-NER",aggregation_strategy="simple")
    keep_list = ['B_animal', 'B_human'] # only relavent ones from animacy_results

    #do parses
    # TODO: edit so that we are not making unnecesarry calls to the API


    # animacy_results = parse_by_pipe(sentences, pipe_animacy,keep_list)
    # ner_results = parse_by_pipe(sentences, pipe_ner)
    animacy_results = parse_by_pipe_splitroles_only(sentences, amr_print, pipe_animacy,keep_list)
    ner_results = parse_by_pipe_splitroles_only(sentences, amr_print, pipe_ner)
    wiki_results = wiki_type(sentences,amr_print)
    pn_results = parse_for_pronouns(sentences)
    ans = combine_parses([ner_results,animacy_results,pn_results,wiki_results])


    return ans


def animacy_decider(X_tuples):
    # I need to create a data structure like this but instead I am feeling it with animate / inaniamate decision
    animacy_info = []
    count = 0
    for _, x_tuple in X_tuples.iterrows():
        # if x_tuple["file"] != file_id:
        #     continue  # Skip if the file ID does not match

        count+=1
        # getting info from X_tuple
        print("\nSentence ", count, ":\n")
        print("sent: \n", x_tuple["sent"])
        # the ne_info is going to have to come from parse_animacy_runner
        print("amr_graph: \n", x_tuple["amr_graph"])
        print("amr_head_name: \n", x_tuple["amr_head_name"])
        print("amr_role: \n", x_tuple["amr_role"])
        print("amr_tail_name: \n", x_tuple["amr_tail_name"])

        # getting each piece of info from X_tuple
        sentence = x_tuple["sent"]
        amr_graph = x_tuple["amr_graph"]
        role = x_tuple["amr_role"]
        tail = x_tuple["amr_tail_name"]
        amr_print = x_tuple["amr_prints"]

        if role == ":mod":
            animacy_info.append("inanimate")
            continue

        # Get ne_info and amr_prints for the current tuple
        # these might end up being wrong we shall see
        print("count:", count)
        named_entity = x_tuple["ne_info"]
        # print("length of ne_info:", len(ne_info))
        # named_entity = ne_info[count - 1]  # Adjust count to start from 0 index
        

        print("named entity: \n", named_entity)
        print("amr_print: \n", amr_print)

        # check for cause-01
        if tail == "cause-01":
            # update the tail
            tail, role = fixCause(amr_graph, tail, role)
        

        decision_animacy = ne_animacy(named_entity, tail, amr_graph, amr_print)
        
        print("ANIMACY DECISION: \n", decision_animacy)
        # Store the animacy decision as a tuple with sentence index and decision
        # animacy_info.append((x_tuple["sent_i"], decision_animacy))
        animacy_info.append(decision_animacy)


    print("INFORMATION IN ANIMACY INFO", animacy_info)
    print("animacy info size: ", len(animacy_info))
    return animacy_info



def parse_animacy_RULES(sentences):
    ##input should be a list, either a list of sentences, a list containing words (could be one word)
    #returns list of dicts that contain combined aimacy, pronoun, and ner info

    #create pipelines
    pipe_animacy = pipeline("token-classification", model="andrewt-cam/bert-finetuned-animacy",aggregation_strategy="simple")
    pipe_ner = pipeline("token-classification", model="dslim/bert-base-NER",aggregation_strategy="simple")
    keep_list = ['B_animal', 'B_human'] # only relavent ones from animacy_results

    #do parses
    # TODO: edit so that we are not making unnecesarry calls to the API
    animacy_results = parse_by_pipe(sentences, pipe_animacy,keep_list)
    ner_results = parse_by_pipe(sentences, pipe_ner)
    pn_results = parse_for_pronouns(sentences)
    ans = combine_parses([ner_results,animacy_results,pn_results])
    return ans


# see if we can mitigate calls to animacy runner
def ne_animacy(named_entity, tail, amr_graph, amr_print):

    # Check if named_entity matches tail
    animacy = animacy_classification(named_entity, tail)
    if animacy != "none":
        return animacy

    # just check the animacy of the tail now if we could not find anything
    print("sending tail to be parsed:", tail)
    new_named_entity = parse_animacy_runner([tail],[amr_print]) # needs to send sentence and tail
    print("NEW NAMED ENTITY: ", new_named_entity)
    animacy = animacy_classification_second_pass(new_named_entity, tail)
    if animacy != "none":
        print("used new animacy identified from running animacy parser again")
        return animacy

    # Run through a second time on the next entity if it was a verb
    animacy = second_pass_animacy(named_entity, tail, amr_graph,amr_print)
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
                if ne_type in ["PER", "B_human", "B_animal","W_Animate"]:
                    print("returning animate!!")
                    return "animate"
                elif ne_type in ["ORG", "LOC", "MISC","W_Inanimate"]:
                    print("returning inanimate!!")
                    return "inanimate"
    return "none"

def animacy_classification_second_pass(named_entity, tail):
    print("in second pass animacy")
    # Check if named_entity is an empty list
    if not named_entity or all(not entry for entry in named_entity):
        print("went to none for second pass animacy")
        return "none"

    # Check if named_entity matches tail
    for ne_entry_list in named_entity:
        for ne_entry in ne_entry_list:
            # Handle the case where named_entity is a string
            if isinstance(ne_entry, str) and ne_entry == tail: # maybe add .lower() here?
                # Check animacy based on ne_type
                if ne_type in ["PER", "B_human", "B_animal"]:
                    return "animate"
                elif ne_type in ["ORG", "LOC", "MISC"]:
                    return "inanimate"
            # Handle the case where named_entity is a dictionary
            elif isinstance(ne_entry, dict):
                for ne_type, ne_value in ne_entry.items():
                    if ne_value == tail:
                        # Check animacy based on ne_type
                        if ne_type in ["PER", "B_human", "B_animal"]:
                            return "animate"
                        elif ne_type in ["ORG", "LOC", "MISC"]:
                            return "inanimate"

    return "none"

def second_pass_animacy(named_entity, tail, amr_graph,amr_print):
    # checking the next node if it is -## assuming it is child of verb
    print("\nin second pass animacy\n")
    if re.search(r'-\d+$', tail):
        # get the label name for find replacement
        role_id = get_label_name(amr_graph, tail)
        # check for the child of this node that is not a verb
        second_check_node = find_replacement_node(amr_graph, tail, role_id)
        print("second check node: ", second_check_node)
        # use the child node in the new animacy classification
        second_animacy = animacy_classification_second_pass(named_entity, second_check_node)
        print("second animacy term: ", second_animacy)
        # if there was no animacy run the parse animacy runner again on the child node
        if second_animacy == "none":
            new_named_entity = parse_animacy_runner([second_check_node],[amr_print])
            # Extract the first dictionary from the list, if any
            if new_named_entity and isinstance(new_named_entity[0], dict):
                new_named_entity = new_named_entity[0]
                print("change to new_named_entity (second pass):", new_named_entity)
            else:
                new_named_entity = {}  # Default to an empty dictionary if no valid dictionary is found
            print("new_named_entity: ",new_named_entity)
            # then return the new animacy classification
            return animacy_classification_second_pass(new_named_entity, second_check_node)
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

def get_label_name(amr_graph, node_id):
    # Find the label name associated with a node ID
    for node in amr_graph.nodes(data="name"):
        if node_id in node:
            return node[0]
    return None

sentences = [
"He showed the sea to the girl.",
"It's extremely troublesome to get there via land.",
"She heated the oven to 250 degrees Celsius.",
"He drove west, from Houston to Austin.",
"I drove to the store.",
"I walked up to the window.",
"The MiG-25 fired an AAM at the Predator.",
"For their honeymoon, the couple flew first class to Hawaii.",
"I showed the pictures to her.",
"He gave the cat some wet food.",
"The boy murmured softly to soothe the girl, because he worried about her.",
"The old man died of lung cancer.",
"The match has been canceled due to the rain.",
"He was injured and therefore unable to play.",
"I can't do work at home because she shouts at me.",
"The hospital has been vacated due to extensive damage.",
"The earthquake caused a tsunami.",
"Scores of people drowned when the boat sank.",
"I must stop now as the shuttle leaves in 10 minutes.",
"John, please -- there are children in the audience.",
"She divorced him in part due to his alcoholism.",
"John Smith, 30, blond, blue eyes, 6'2, 200 lbs.",
"From 1953 to 1955, 9.8 billion Kent cigarettes with the filters were sold, the company said.",
"Of course, VDOT has no more money for road construction in the Richmond region.",
"Regarding foreign contracted projects and cooperation of labor service, the Hong Kong region is still the most important market of the inland.",
"It applies to male adherents of feminism too, you know, limp-wristed, leftist men.",
"Texas, Especially Austin, Needs Help!",
"Among common birds, a rather special one is the black-faced spoonbill.",
"They collect and prepare nonperishable food boxes for local shelters and meal programs, and are helping with the Katrina effort.",
"Tornado rakes Southern Indiana; Marysville, town of 1,900, 'completely gone' | The Indianapolis Star | indystar.com",
"The Monkey came with the rest and presented, with all a mother's tenderness, a flat-nosed, hairless, ill-featured young Monkey as a candidate for the promised reward.",
"I saw a cloud of dust.",
"A team of researchers made a discovery.",
"A ring of gold.",
"The children's choir sang beautifully.",
"The eXchanger Inhibitory Peptide (XIP) region (residues 251-271) of NCX1.",
"Workers described 'clouds of blue dust' that hung over parts of the factory, even though exhaust fans ventilated the area.",
"Rush, Levin, Hannity, Savage etc all said Romney was the conservative candidate.",
"South Koreans rallied on January 11, 2002, in support of their right to eat dog meat.",
"Religious extremism continues in Pakistan despite the banning of militant groups.",
"They will focus on the import of cheap chemical drugs made in Thailand.",
"The boy met a girl from Spain.",
"He drove west, from Houston to Austin.",
"I backed away from the window.",
"These have all been vacated too from what I'm hearing.",
"Physicists from all over the world.",
"She got a master's degree in linguistics from UCLA.",
"Hallmark could make a fortune off of this guy.",
"It stimulates others to be big contributors so they may be in on the next wave of free cash from the boy king.",
"The Wonder Tour will start from Hong Kong.",
"After all, what are the joyful memories from inside paradise?",
"Then welcome to the official writing ceremony of Hong Kong Disneyland.",
"This dynamic metropolis never ceases its movement.",
"According to government sources, the killing happened yesterday.",
"I ate pasta with tomato sauce.",
"A special gubernatorial election will take place next Tuesday.",
"Microbial virus.",
"First let's focus on the main contents of today's program.",
"Well, originally the construction of the new --",
"Establishing Models in Industrial Innovation.",
"In addition, there was something else that was very suspicious."
]

#word = ["shuttle", "the shuttle is cold"]

#print(parse_animacy_runner(sentences))
#print(parse_animacy_runner(word))