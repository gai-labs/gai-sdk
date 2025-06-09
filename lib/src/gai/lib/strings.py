import re

def clean_string(s):
    if s is None:
        return ''
    return re.sub(r'\s+', ' ', s)
