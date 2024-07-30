import re

def extract_information(string):
    if '.<|im_end|>' in string:
        string = string.split('.<|im_end|>')[0]+"."
    string = string.replace('<|im_end|>\n','')
    match = re.search(r'\n(.*?)\n', string, re.DOTALL)
    if match:
        string = match.group(1)
    string = string.replace('<|endoftext|>','')
    string = string.strip()
    return string

x = "Lesion<|endoftext|><|endoftext|>"
print(extract_information(x))