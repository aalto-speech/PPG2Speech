import re

def remove_punc_and_tolower(s: str) -> str:
    return re.sub(r'[^\w\s]', '', s).lower()