# import spacy
import re


class peptokenize(object):
    def tokenizer(self, sentence):
        return [tok for tok in (sentence) if tok != " "]

