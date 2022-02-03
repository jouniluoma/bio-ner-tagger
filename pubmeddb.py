#!/usr/bin/env python3

import spacy
from collections import namedtuple
import sys
import re

import numpy as np

from common import process_sentences, encode
from collections import deque

PubMedDocument = namedtuple(
    'PubMedDocument',
    'doc_id, text'
)


PubMedSpan = namedtuple(
    'PubMedSpan',
    'doc_id, par_num, type_id, start, end, text'
)


# Alnum sequences preserved as single tokens, rest are
# single-character tokens.
TOKENIZATION_RE = re.compile(r'([^\W_]+|.)')

PubMedDoc = namedtuple(
    'PubMedDoc',
    'doc_id, text, tids, sids, data'
)


# def sentence_split_old(text):
#     if sentence_split.nlp is None:
#         # Cache spacy model
#         nlp = spacy.load('en_core_sci_sm', disable=['tagger', 'ner','pos'])
#         nlp.add_pipe('sentencizer')
#         sentence_split.nlp = nlp
#     return [s.text for s in sentence_split.nlp(text).sents]


def sentence_split(text):
    if sentence_split.nlp is None:
        # Cache spacy model                                                     
        nlp = spacy.load('en_core_sci_sm', disable=['tagger','parser','ner','lemmatizer','textcat','pos'])
        nlp.add_pipe('sentencizer')
        sentence_split.nlp = nlp
    sentence_texts = []
    for para in text.split('\t'):
        sentence_texts.extend([s.text for s in sentence_split.nlp(para).sents])
    return sentence_texts
sentence_split.nlp = None

def tokenize(text):
    return [t for t in TOKENIZATION_RE.split(text) if t and not t.isspace()]


def split_and_tokenize(text):
    sentences = sentence_split(text)
    return [tokenize(s) for s in sentences]


def dummy_labels(tokenized_sentences):
    sentence_labels = []
    for tokens in tokenized_sentences:
        sentence_labels.append(['O'] * len(tokens))
    return sentence_labels


def get_word_labels(orig_words, token_lengths, tokens, predictions):
    """Map wordpiece token labels to word labels."""
    toks = deque([val for sublist in tokens for val in sublist])
    pred = deque([val for sublist in predictions for val in sublist])
    lengths = deque(token_lengths)
    word_labels = []
    for sent_words in orig_words:
        sent_labels = []
        for _ in sent_words:
            sent_labels.append(pred.popleft())
            for i in range(int(lengths.popleft())-1):
                pred.popleft()
        word_labels.append(sent_labels)
    return word_labels


def iob2_span_ends(curr_type, tag):
    if curr_type is None:
        return False
    elif tag == 'I-{}'.format(curr_type):
        return False
    elif tag == 'O' or tag[0] == 'B':
        return True
    else:
        assert curr_type != tag[2:], 'internal error'
        return True    # non-IOB2 or tag sequence error


def iob2_span_starts(curr_type, tag):
    if tag == 'O':
        return False
    elif tag[0] == 'B':
        return True
    elif curr_type is None:
        return True    # non-IOB2 or tag sequence error
    else:
        assert tag == 'I-{}'.format(curr_type), 'internal error'
        return False


def tags_to_spans(text, tokens, tags):
    spans = []
    offset, curr_type, start = 0, None, None
    assert len(tokens) == len(tags)
    for token, tag in zip(tokens, tags):
        if iob2_span_ends(curr_type, tag):
            spans.append((start, offset, curr_type, text[start:offset]))
            curr_type, start = None, None
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if text[offset:offset+len(token)] != token:
            raise ValueError('text mismatch')
        if iob2_span_starts(curr_type, tag):
            curr_type, start = tag[2:], offset
        offset += len(token)
    if curr_type is not None:
        spans.append((start, offset, curr_type, text[start:offset]))
    return spans

def write_sentences(outfile, sentences, labels):
    for sentence, tagseq in zip(sentences,labels):
        for word, tag in zip(sentence, tagseq):
            outfile.write('{}\t{}\n'.format(word, tag))
        outfile.write('\n')

def writespans(infile, doc_id, spans):
    for i,s in enumerate(spans):
        infile.write('{}\tT{}\t{}\t{}\t{}\t{}\n'.format(doc_id,i+1,s[2],s[0],s[1],s[3]))



def create_samples(tokenizer, seq_len, document):
    words = split_and_tokenize(document.text)
    labels = dummy_labels(words)
    data = process_sentences(words, labels, tokenizer, seq_len) #One doc at time --> documentwise
    tids, sids = encode(data.combined_tokens, tokenizer, seq_len)
    return PubMedDoc(document.doc_id, document.text, tids, sids, data)    



def parse_PubMeddb_input_line(line):
    """Parse line in database_documents.tsv format, return PubMedDocument."""
    line = line.rstrip('\n')
    fields = line.split('\t')
    doc_id, heading, text = fields
    text = heading+'\n'+text
    return PubMedDocument(doc_id, text)


def stream_documents(fn):
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            try:
                document = parse_PubMeddb_input_line(l)
            except Exception as e:
                raise ValueError('failed to parse {} line {}'.format(fn, ln))
            yield document


if __name__ == '__main__':
    import sys

    # Test I/O
    for fn in sys.argv[1:]:
        for doc in stream_documents(fn):
            print(doc.doc_id, len(doc.text.split()), 'tokens')