# Use a pipeline as a high-level helper


pronouns = ['i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them']

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
    return list_of_parses[0]


import string
def remove_punctuation(input_string):
    # Create a translation table for removing punctuation
    translator = str.maketrans('', '', string.punctuation)
    
    # Apply the translation to the input string
    result = input_string.translate(translator)
    
    return result



#parse_for_animacy("My name is Clara and I live in Berkeley, California.", pipe_ner)
#pn  = parse_for_pronouns("My name is Clara and I live in Berkeley, California.",pronouns)


