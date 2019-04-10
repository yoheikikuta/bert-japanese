# coding=utf-8
# This file is based on https://github.com/google-research/bert/blob/master/tokenization.py.
# It is changed to use SentencePiece tokenizer for tokenizations.
"""Tokenization classes."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import unicodedata
import sentencepiece as sp
import six
import tensorflow as tf


def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
    """Checks whether the casing config is consistent with the checkpoint name."""

    # The casing has to be passed in by the user and there is no explicit check
    # as to whether it matches the checkpoint. The casing information probably
    # should have been stored in the bert_config.json file, but it's not, so
    # we have to heuristically detect it to validate.

    if not init_checkpoint:
        return

    m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
    if m is None:
        return

    model_name = m.group(1)

    lower_models = [
        "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
        "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
    ]

    cased_models = [
        "cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
        "multi_cased_L-12_H-768_A-12"
    ]

    is_bad_config = False
    if model_name in lower_models and not do_lower_case:
        is_bad_config = True
        actual_flag = "False"
        case_name = "lowercased"
        opposite_flag = "True"

    if model_name in cased_models and do_lower_case:
        is_bad_config = True
        actual_flag = "True"
        case_name = "cased"
        opposite_flag = "False"

    if is_bad_config:
        raise ValueError(
            "You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`. "
            "However, `%s` seems to be a %s model, so you "
            "should pass in `--do_lower_case=%s` so that the fine-tuning matches "
            "how the model was pre-training. If this error is wrong, please "
            "just comment out this check." % (actual_flag, init_checkpoint,
                                              model_name, case_name, opposite_flag))


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token, _ = token.split("\t")
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def convert_by_vocab(vocab, items, unk_info):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        if item in vocab:
            output.append(vocab[item])
        else:
            output.append(unk_info)
    return output


def convert_tokens_to_ids(vocab, tokens):
    """Id of <unk> is assumed as 0 accroding to sentencepiece"""
    return convert_by_vocab(vocab, tokens, unk_info=0)


def convert_ids_to_tokens(inv_vocab, ids):
    """Token of unknown word is assumed as <unk> according to sentencepiece"""
    return convert_by_vocab(inv_vocab, ids, unk_info="<unk>")


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, model_file, vocab_file, do_lower_case):
        self.tokenizer = SentencePieceTokenizer(model_file, do_lower_case=do_lower_case)
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        split_tokens = self.tokenizer.tokenize(text)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Id of <unk> is assumed as 0 accroding to sentencepiece"""
        return convert_by_vocab(self.vocab, tokens, unk_info=0)

    def convert_ids_to_tokens(self, ids):
        """Token of unknown word is assumed as <unk> according to sentencepiece"""
        return convert_by_vocab(self.inv_vocab, ids, unk_info="<unk>")


class SentencePieceTokenizer(object):
    
    nmt_norm_map = str.maketrans({
            # SPACES
            '\u0009':'\u0020',  # TAB
            '\u000A':'\u0020',  # LINE FEED
            '\u000C':'\u0020',  # FORM FEED
            '\u000D':'\u0020',  # CARRIAGE RETURN
            '\u1680':'\u0020',  # OGHAM SPACE MARK
            '\u200B':'\u0020',  # ZERO WIDTH SPACE
            '\u200E':'\u0020',  # LEFT-TO-RIGHT MARK
            '\u200F':'\u0020',  # RIGHT-TO-LEFT MARK
            '\u2028':'\u0020',  # LINE SEPARATOR
            '\u2029':'\u0020',  # PARAGRAPH SEPARATOR
            '\u2581':'\u0020',  # LOWER ONE EIGHT BLOCK
            '\uFEFF':'\u0020',  # ZERO WIDTH NO-BREAK
            '\uFFFD':'\u0020',  # REPLACEMENT CHARACTER
            '\u200C':'\u0020',  # ZERO WIDTH NON-JOINER
            '\u200D':'\u0020',  # ZERO WIDTH JOINER
            
            # Ascii Control characters
            '\u0001':'',
            '\u0002':'',
            '\u0003':'',
            '\u0004':'',
            '\u0005':'',
            '\u0006':'',
            '\u0007':'',
            '\u0008':'',
            '\u000B':'',
            '\u000E':'',
            '\u000F':'',
            '\u0010':'',
            '\u0011':'',
            '\u0012':'',
            '\u0013':'',
            '\u0014':'',
            '\u0015':'',
            '\u0016':'',
            '\u0017':'',
            '\u0018':'',
            '\u0019':'',
            '\u001A':'',
            '\u001B':'',
            '\u001C':'',
            '\u001D':'',
            '\u001E':'',
            '\u001F':'',
            
            #  <control-007F>..<control-009F>
            '\u007F':'',
            '\u008F':'',
            '\u009F':'',    
        })
    
    @staticmethod
    def normalize_with_nmt_NFKC(
            text, 
            treat_whitespace_as_suffix_=False, 
            add_dummy_prefix=True, 
            remove_extra_whitespaces=True, 
            escape_whitespaces=True,
            do_lower_case=True,
        ):
        """
        An emulation of sp normalizer with nmt NFKC
        This method is not required before inputing tokens into the sp tokenizer because it normalize them by itself
        You can know in advance what the whole string of the tokenized text will be
        """
        # custom mapping for nmt
        text = text.translate(SentencePieceTokenizer.nmt_norm_map)
        # tilde protection (storing)
        tildes = filter(lambda c: c == '\uFF5E' or c == '\u007E', text)
        # nfkc normalization
        text = unicodedata.normalize('NFKC', text)
        # tilde protection (restoring)
        text = ''.join([c if c != '\u007E' else next(tildes) for c in text])
        
        # triming extra spaces
        if remove_extra_whitespaces:
            text = re.sub('\u0020+', '\u0020', text.strip())
        # dummy space
        if add_dummy_prefix:
            if treat_whitespace_as_suffix_:
                text = text + '\u0020'
            else:
                text = '\u0020' + text
        # escaping spaces
        if escape_whitespaces:
            text = text.replace('\u0020', '\u2581')
        
        # do_lower_case which is a part of BERT script
        if do_lower_case:
            text = text.lower()
        return text
    
    def __init__(self, model_file=None, independent_tokens=None, do_lower_case=True):
        """Constructs a SentencePieceTokenizer."""
        self.tokenizer = sp.SentencePieceProcessor()
        if self.tokenizer.Load(model_file):
            print("Loaded a trained SentencePiece model.")
        else:
            print("You have to give a path of trained SentencePiece model.")
            sys.exit(1)
        
        self.do_lower_case = do_lower_case
        
        self.independent_tokens = independent_tokens
        if self.independent_tokens is None:
            self.independent_tokens = ['、', '\u2581']
    
    def tokenize(self, text, normalize=False):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        if normalize:
            text = SentencePieceTokenizer.normalize_with_nmt_NFKC(text)
        if self.do_lower_case:
            text = text.lower()
        
        output_tokens = self.tokenizer.EncodeAsPieces(text)
        output_tokens = self.output_hook(output_tokens)
        return output_tokens

    def output_hook(self, tokens):
        """
        1. separete some prefix (specified in self.independent_tokens) in tokens
        2. remove "\u2581" at the first position inserted due to the add_dummy_prefix flag.
        """
        
        #affected = []
        #def _write():
        #    with open('affected_tokens.txt', 'a') as f:
        #        f.write(str(affected))
        #        f.write('\n')
        
        if len(tokens) == 0:
            #_write()
            return []
            
        if tokens[0].startswith('\u2581'):
            #if len(tokens[0]) > 1:
            #    affected.append(tokens[0])
            tokens[0] = tokens[0][1:]
            if len(tokens[0]) == 0:
                del tokens[0]
                if len(tokens) == 0:
                    #_write()
                    return []
        
        def is_separatable_prefix(t):
            if len(t) > 1 and t[0] in self.independent_tokens:
                return True
            return False
        
        new_tokens = []
        
        for i in range(len(tokens)):
            token = tokens[i]
            while is_separatable_prefix(token):    
                new_tokens.append(token[0])
                token = token[1:]
            new_tokens.append(token)
            #if token != tokens[i]:
            #    affected.append(tokens[i])
        
        #_write()
        return new_tokens
    
