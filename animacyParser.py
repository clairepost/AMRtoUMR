# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("token-classification", model="andrewt-cam/bert-finetuned-animacy")

def parse_for_animacy(input_sent):
    animacy_info = pipe(input_sent)
    print(animacy_info)
    print(animacy_info, file = open("animacy_parse.txt","w"))
    return

parse_for_animacy("My name is Clara and I live in Berkeley, California.")