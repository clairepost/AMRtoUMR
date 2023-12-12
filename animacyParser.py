# Use a pipeline as a high-level helper
import string
from transformers import pipeline


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



def remove_punctuation(input_string):
    # Create a translation table for removing punctuation
    translator = str.maketrans('', '', string.punctuation)
    
    # Apply the translation to the input string
    result = input_string.translate(translator)
    
    return result




def parse_animacy_runner(sentences):
    ##input should be a list, either a list of sentences, a list containing words (could be one word)
    #returns list of dicts that contain combined aimacy, pronoun, and ner info
    print("testing animacy on", len(sentences), "sentences/words")
    #create pipelines
    pipe_animacy = pipeline("token-classification", model="andrewt-cam/bert-finetuned-animacy",aggregation_strategy="simple")
    pipe_ner = pipeline("token-classification", model="dslim/bert-base-NER",aggregation_strategy="simple")
    keep_list = ['B_animal', 'B_human'] # only relavent ones from animacy_results

    #do parses
    animacy_results = parse_by_pipe(sentences, pipe_animacy,keep_list)
    ner_results = parse_by_pipe(sentences, pipe_ner)
    pn_results = parse_for_pronouns(sentences)
    ans = combine_parses([ner_results,animacy_results,pn_results,])
    return ans


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

word = ["children"]

print(parse_animacy_runner(sentences))
print(parse_animacy_runner(word))