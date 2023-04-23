import json
import os

import joblib
import numpy as np
from ovos_classifiers.skovos.classifier import SklearnOVOSClassifier
from ovos_utils.xdg_utils import xdg_data_home
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC


class StrEnum:
    """Enum with strings as keys. Implements items method"""

    @classmethod
    def values(cls):
        return [getattr(cls, i) for i in dir(cls)
                if not i.startswith("__") and i != 'values']


class IdManager:
    """
    Gives manages specific unique identifiers for tokens.
    Used to convert tokens to vectors
    """

    def __init__(self, id_cls=StrEnum, ids=None):
        if ids is not None:
            self.ids = ids
        else:
            self.ids = {}
            for i in id_cls.values():
                self.add_token(i)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def adj_token(token):
        if token.isdigit():
            for i in range(10):
                if str(i) in token:
                    token = token.replace(str(i), '#')
        return token

    def vector(self):
        return [0.0] * len(self.ids)

    def save(self, prefix):
        with open(prefix + '.ids', 'w') as f:
            json.dump(self.ids, f)

    def load(self, prefix):
        with open(prefix + '.ids', 'r') as f:
            self.ids = json.load(f)

    def assign(self, vector, key, val):
        vector[self.ids[self.adj_token(key)]] = val

    def __contains__(self, token):
        return self.adj_token(token) in self.ids

    def add_token(self, token):
        token = self.adj_token(token)
        if token not in self.ids:
            self.ids[token] = len(self.ids)

    def add_sent(self, sent):
        for token in sent:
            self.add_token(token)


class Ids(StrEnum):
    unknown_tokens = ':0'
    w_1 = ':1'
    w_2 = ':2'
    w_3 = ':3'
    w_4 = ':4'


class PIds(StrEnum):
    end = ':end'


class PadatiousEntityVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, direction, token):
        super().__init__()
        self.ids = IdManager(PIds)
        self.token = token
        self.direction = direction

    def get_end(self, sent):
        return len(sent) if self.direction > 0 else -1

    def vectorize(self, tokens, idx):
        unknown = 0
        vector = self.ids.vector()
        end_pos = self.get_end(tokens)
        for i in range(idx + self.direction, end_pos, self.direction):
            if tokens[i] in self.ids:
                self.ids.assign(vector, tokens[i], 1.0 / abs(i - idx))
            else:
                unknown += 1
        self.ids.assign(vector, PIds.end, 1.0 / abs(end_pos - idx))
        return vector

    def fit(self, X, **kwargs):
        if self.token in X:
            for i in range(X.index(self.token) + self.direction,
                           self.get_end(X), self.direction):
                if X[i][0] != '{':
                    self.ids.add_token(X[i])
        return self

    def transform(self, X, **transform_params):
        feats = [self.vectorize(X, index) for index in range(len(X))]
        return feats


class PadatiousVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.ids = IdManager(Ids)
        self.pos_intents = {}

    def save(self, path):
        joblib.dump(self, path + ".ids")

    def load_from_file(self, path):
        vectr = joblib.load(path + ".ids")
        self.ids = vectr.ids
        self.pos_intents = vectr.pos_intents
        return self

    def vectorize(self, tokens):
        vector = self.ids.vector()
        unknown = 0
        for token in tokens:
            if token in self.ids:
                self.ids.assign(vector, token, 1.0)
            else:
                unknown += 1
        if len(tokens) > 0:
            self.ids.assign(vector, Ids.unknown_tokens, unknown / float(len(tokens)))
            self.ids.assign(vector, Ids.w_1, len(tokens) / 1)
            self.ids.assign(vector, Ids.w_2, len(tokens) / 2.)
            self.ids.assign(vector, Ids.w_3, len(tokens) / 3.)
            self.ids.assign(vector, Ids.w_4, len(tokens) / 4.)
        return vector

    def fit(self, X, *args, **kwargs):
        for tokens in X:
            self.ids.add_sent(tokens)
            for tok in tokens:
                if tok not in self.pos_intents:
                    self.pos_intents[tok] = [PadatiousEntityVectorizer(+1, tok),
                                             PadatiousEntityVectorizer(-1, tok)]
                self.pos_intents[tok][0].fit(tokens)
                self.pos_intents[tok][1].fit(tokens)
        return self

    def transform(self, X, *args, **transform_params):
        return [self.vectorize(x) for x in X]


class PadatiousSklearn(SklearnOVOSClassifier):
    LENIENCE = 0.6
    ALGO = SVC

    def __init__(self):
        super().__init__("padatious", self.ALGO(probability=True))
        # convert y values to categorical values
        self.encoder = preprocessing.LabelEncoder()

    def predict(self, text):
        return np.max(self.clf.predict_proba(text), axis=1)

    def save(self, path):
        joblib.dump(self.encoder, path + ".encoder")
        super().save(path)

    def load_from_file(self, path=None):
        if not path:
            os.makedirs(f"{xdg_data_home()}/OpenVoiceOS/classifiers", exist_ok=True)
            path = f"{xdg_data_home()}/OpenVoiceOS/classifiers/{self.pipeline_id}"
        self.encoder = joblib.load(path + ".encoder")
        return super().load_from_file(path)

    @property
    def pipeline(self):
        return [
            ('feats', PadatiousVectorizer()),
            ('clf', self._pipeline_clf)
        ]

    def train(self, train_data, target_data):
        inputs, outputs = self._augment(train_data, target_data)
        outputs = self.encoder.fit_transform(outputs)
        return super().train(inputs, outputs)

    def score(self, X, y):
        y = self.encoder.fit_transform(y)
        return super().score(X, y)

    @classmethod
    def _augment(cls, train_data, target_data, entities=None):
        entities = entities or {}

        for idx, intent in enumerate(target_data):
            toks = train_data[idx]
            for j, check_token in enumerate(toks):
                d = j - idx
                if int(d > 0) - int(d < 0) in [1, -1] and check_token.startswith('{'):
                    for pol_len in range(1, 4):
                        s = toks[:j] + [':0'] * pol_len + toks[j + 1:]
                        train_data.append(s)
                        target_data.append(1)

        inputs = []
        outputs = []

        def add(toks, out):

            inputs.append(toks)
            outputs.append(out)

        def pollute(tok, p):
            tok = tok[:]
            for _ in range(int((len(tok) + 2) / 3)):
                tok.insert(p, ':null:')
            add(tok, cls.LENIENCE)

        def weight(tok):
            def calc_weight(w):
                return pow(len(w), 3.0)

            total_weight = 0.0
            for word in tok:
                total_weight += calc_weight(word)
            for word in tok:
                weight = 0 if word.startswith('{') else calc_weight(word)
                add([word], weight / total_weight)

        for idx, toks in enumerate(train_data):
            if target_data[idx] == 0:
                add(toks, 0.0)
            else:
                add(toks, 1.0)
                weight(toks)

                # Generate samples with extra unknown tokens unless
                # the sentence is supposed to allow unknown tokens via the special :0
                if not any(word[0] == ':' and word != ':' for word in toks):
                    pollute(toks, 0)
                    pollute(toks, len(toks))

        add([':null:'], 0.0)
        add([], 0.0)

        for entity, samples in entities.items():
            for s in samples:
                for idx, toks in enumerate(train_data):
                    with_entity = toks[:]
                    for i, token in enumerate(with_entity):
                        if token.startswith('{'):
                            ent = token.lstrip("{").rstrip("}")
                            if ent == entity:
                                with_entity[i] = s
                    if with_entity != toks:
                        add(with_entity, 1.0)

        for idx, toks in enumerate(train_data):
            if target_data[idx] == 0:
                continue
            without_entities = toks[:]
            for i, token in enumerate(without_entities):
                if token.startswith('{'):
                    without_entities[i] = ':null:'
            if without_entities != toks:
                add(without_entities, 0.0)

        return inputs, outputs

    @classmethod
    def intent2dataset(cls, X, y, target_intent, entities=None):
        inputs = []
        outputs = []

        for idx, intent in enumerate(y):
            toks = X[idx].split()
            inputs.append(toks)
            if intent != target_intent:
                outputs.append(0)
            else:
                outputs.append(1)

        return cls._augment(inputs, outputs, entities)


