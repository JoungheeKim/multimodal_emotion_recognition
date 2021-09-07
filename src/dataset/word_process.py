from konlpy.tag import Mecab
from sklearn.preprocessing import normalize
from soynlp.hangle import decompose, character_is_korean
import numpy as np

class Word2Vec:
    def __init__(self, vecs_txt_fname, vecs_bin_fname=None, method="word2vec", tokenizer_name="mecab"):
        self.tokenizer = get_tokenizer(tokenizer_name)
        self.tokenizer_name = tokenizer_name
        self.method = method
        self.dictionary, self.words, self.vecs = self.load_vectors(vecs_txt_fname, method)
        self.dim = self.vecs.shape[1]
        if "fasttext" in method:
            from fasttext import load_model as load_ft_model
            self.model = load_ft_model(vecs_bin_fname)

    def get_word_vector(self, word):
        if self.method == "fasttext-jamo":
            word = jamo_sentence(word)
        if self._is_in_vocabulary(word):
            vector = self.dictionary[word]
        else:
            if "fasttext" in self.method:
                vector = self.model.get_word_vector(word)
            else:
                vector = np.zeros(self.dim)
                print("설마 여기인가?")
        return vector

    def load_vectors(self, vecs_fname, method):
        if method == "word2vec":
            from gensim.models import Word2Vec
            model = Word2Vec.load(vecs_fname)
            ## gensim fuction 'index2word' is replaced to 'index_to_key'
            ## words = model.wv.index2word
            words = model.wv.index_to_key
            vecs = model.wv.vectors
        else:
            words, vecs = [], []
            with open(vecs_fname, 'r', encoding='utf-8') as f:
                if "fasttext" in method:
                    next(f)  # skip head line
                for line in f:
                    if method == "swivel":
                        splited_line = line.strip().split("\t")
                    else:
                        splited_line = line.strip().split(" ")
                    words.append(splited_line[0])
                    vec = [float(el) for el in splited_line[1:]]
                    vecs.append(vec)
        unit_vecs = normalize(vecs, norm='l2', axis=1)
        dictionary = {}
        for word, vec in zip(words, unit_vecs):
            dictionary[word] = vec
        return dictionary, words, unit_vecs

    def tokenize(self, sentence: str):
        sentence = sentence.strip()
        if self.tokenizer_name == "khaiii":
            tokens = []
            for word in self.tokenizer.analyze(sentence):
                tokens.extend([str(m).split("/")[0] for m in word.morphs])
        else:
            tokens = self.tokenizer.morphs(sentence)
        return tokens

    def encode(self, data):
        assert type(data) in [str, list], 'not supported data-type [{}]'.format(type(data))

        output = None
        if type(data) == str:
            sentence = data
            tokens = self.tokenize(sentence)
            encoded_tokens = [self.get_word_vector(token) for token in tokens]
            output = np.array(encoded_tokens)

        if type(data) == list:
            output = list()
            for sentence in data:
                tokens = self.tokenize(sentence)
                encoded_tokens = [self.get_word_vector(token) for token in tokens]
                encoded_tokens = np.array(encoded_tokens)
                output.append(encoded_tokens)
        return output

    def post_processing(self, tokens):
        results = []
        for token in tokens:
            # 숫자에 공백을 주어서 띄우기
            processed_token = [el for el in re.sub(r"(\d)", r" \1 ", token).split(" ") if len(el) > 0]
            results.extend(processed_token)
        return results

    def jamo_to_word(self, jamo):
        jamo_list, idx = [], 0
        while idx < len(jamo):
            if not character_is_korean(jamo[idx]):
                jamo_list.append(jamo[idx])
                idx += 1
            else:
                jamo_list.append(jamo[idx:idx + 3])
                idx += 3
        word = ""
        for jamo_char in jamo_list:
            if len(jamo_char) == 1:
                word += jamo_char
            elif jamo_char[2] == "-":
                word += compose(jamo_char[0], jamo_char[1], " ")
            else:
                word += compose(jamo_char[0], jamo_char[1], jamo_char[2])
        return word

    def _is_in_vocabulary(self, word):
        if self.method == "fasttext-jamo":
            word = jamo_sentence(word)
        return word in self.dictionary.keys()


def get_tokenizer(tokenizer_name):
    from konlpy.tag import Okt, Komoran, Mecab, Hannanum, Kkma
    if tokenizer_name == "komoran":
        tokenizer = Komoran()
    elif tokenizer_name == "okt":
        tokenizer = Okt()
    elif tokenizer_name == "mecab":
        tokenizer = Mecab()
    elif tokenizer_name == "hannanum":
        tokenizer = Hannanum()
    elif tokenizer_name == "kkma":
        tokenizer = Kkma()
    elif tokenizer_name == "khaiii":
        from khaiii import KhaiiiApi
        tokenizer = KhaiiiApi()
    else:
        tokenizer = Mecab()
    return tokenizer


from soynlp.hangle import decompose, character_is_korean
import re

doublespace_pattern = re.compile('\s+')


def jamo_sentence(sent):
    def transform(char):
        if char == ' ':
            return char
        cjj = decompose(char)
        if len(cjj) == 1:
            return cjj
        cjj_ = ''.join(c if c != ' ' else '-' for c in cjj)
        return cjj_

    sent_ = []
    for char in sent:
        if character_is_korean(char):
            sent_.append(transform(char))
        else:
            sent_.append(char)
    sent_ = doublespace_pattern.sub(' ', ''.join(sent_))
    return sent_