# preprocessing.py

import re
import string
import pandas as pd

class TextPreprocessor:
    def __init__(self):
        # Defining the contraction patterns
        self.cont_patterns = [
            (b'(W|w)on\'t', b'will not'),
            (b'(C|c)an\'t', b'can not'),
            (b'(I|i)\'m', b'i am'),
            (b'(A|a)in\'t', b'is not'),
            (b'(\w+)\'ll', b'\g<1> will'),
            (b'(\w+)n\'t', b'\g<1> not'),
            (b'(\w+)\'ve', b'\g<1> have'),
            (b'(\w+)\'s', b'\g<1> is'),
            (b'(\w+)\'re', b'\g<1> are'),
            (b'(\w+)\'d', b'\g<1> would'),
        ]
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in self.cont_patterns]

    @staticmethod
    def prepare_for_char_n_gram(text):
        """ Simple text clean up process"""
        clean = bytes(text.lower(), encoding="utf-8")
        clean = clean.replace(b"\n", b" ")
        clean = clean.replace(b"\t", b" ")
        clean = clean.replace(b"\b", b" ")
        clean = clean.replace(b"\r", b" ")

        for (pattern, repl) in TextPreprocessor.patterns:
            clean = re.sub(pattern, repl, clean)

        exclude = re.compile(b'[%s]' % re.escape(bytes(string.punctuation, encoding='utf-8')))
        clean = b" ".join([exclude.sub(b'', token) for token in clean.split()])
        clean = re.sub(b"\d+", b" ", clean)
        clean = re.sub(b'\s+', b' ', clean)
        clean = re.sub(b'\s+$', b'', clean)
        clean = re.sub(b" ", b"# #", clean)
        clean = b"#" + clean + b"#"

        return str(clean, 'utf-8')

    @staticmethod
    def count_regexp_occ(regexp="", text=None):
        """ Simple way to get the number of occurrences of a regex"""
        return len(re.findall(regexp, text))

    def get_indicators_and_clean_comments(self, df):
        df["ant_slash_n"] = df["comment_text"].apply(lambda x: self.count_regexp_occ(r"\n", x))
        df["raw_word_len"] = df["comment_text"].apply(lambda x: len(x.split()))
        df["raw_char_len"] = df["comment_text"].apply(lambda x: len(x))
        df["nb_upper"] = df["comment_text"].apply(lambda x: self.count_regexp_occ(r"[A-Z]", x))
        df["nb_fk"] = df["comment_text"].apply(lambda x: self.count_regexp_occ(r"[Ff]\S{2}[Kk]", x))
        df["nb_sk"] = df["comment_text"].apply(lambda x: self.count_regexp_occ(r"[Ss]\S{2}[Kk]", x))
        df["nb_dk"] = df["comment_text"].apply(lambda x: self.count_regexp_occ(r"[dD]ick", x))
        df["nb_you"] = df["comment_text"].apply(lambda x: self.count_regexp_occ(r"\W[Yy]ou\W", x))
        df["nb_ng"] = df["comment_text"].apply(lambda x: self.count_regexp_occ(r"\Wnigger\W", x))
        df["start_with_columns"] = df["comment_text"].apply(lambda x: self.count_regexp_occ(r"^\:+", x))
        df["has_timestamp"] = df["comment_text"].apply(lambda x: self.count_regexp_occ(r"\d{2}|:\d{2}", x))
        df["has_date_long"] = df["comment_text"].apply(lambda x: self.count_regexp_occ(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", x))
        df["has_date_short"] = df["comment_text"].apply(lambda x: self.count_regexp_occ(r"\D\d{1,2} \w+ \d{4}", x))
        df["has_http"] = df["comment_text"].apply(lambda x: self.count_regexp_occ(r"http[s]{0,1}://\S+", x))
        df["has_mail"] = df["comment_text"].apply(
            lambda x: self.count_regexp_occ(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x)
        )
        df["has_emphasize_equal"] = df["comment_text"].apply(lambda x: self.count_regexp_occ(r"\={2}.+\={2}", x))
        df["has_emphasize_quotes"] = df["comment_text"].apply(lambda x: self.count_regexp_occ(r"\"{4}\S+\"{4}", x))

        df["clean_comment"] = df["comment_text"].apply(lambda x: TextPreprocessor.prepare_for_char_n_gram(x))

        df["clean_word_len"] = df["clean_comment"].apply(lambda x: len(x.split()))
        df["clean_char_len"] = df["clean_comment"].apply(lambda x: len(x))
        df["clean_chars"] = df["clean_comment"].apply(lambda x: len(set(x)))
        df["clean_chars_ratio"] = df["clean_comment"].apply(lambda x: len(set(x))) / df["clean_comment"].apply(
            lambda x: 1 + min(99, len(x)))
