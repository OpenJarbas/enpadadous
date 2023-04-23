from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC

from enpadadous.clf import EnpadadousSklearn, EnpadadousVotingSklearn
from enpadadous.padaos_engine import PadaosIntentContainer


@dataclass()
class IntentMatch:
    intent_name: str
    confidence: float
    entities: dict


class EnpadadousIntentContainer:
    def __init__(self, clf=None):
        clf = clf or [SVC(probability=True), LogisticRegression(), Perceptron()]
        self.padaos = PadaosIntentContainer()
        if isinstance(clf, list):
            self.enpadadous = EnpadadousVotingSklearn(clf)
        else:
            self.enpadadous = EnpadadousSklearn(clf)
        self.intent_lines, self.entity_lines = {}, {}
        self.intent_clfs = {}

    def add_intent(self, name, lines):
        self.padaos.add_intent(name, lines)
        self.intent_lines[name] = lines

    def remove_intent(self, name):
        self.padaos.remove_intent(name)
        if name in self.intent_lines:
            del self.intent_lines[name]

    def add_entity(self, name, lines):
        self.padaos.add_entity(name, lines)
        self.entity_lines[name] = lines

    def remove_entity(self, name):
        self.padaos.remove_entity(name)
        if name in self.entity_lines:
            del self.entity_lines[name]

    def calc_intents(self, query):
        for exact_intent in self.padaos.calc_intents(query):
            if exact_intent["name"]:
                yield IntentMatch(confidence=1.0,
                                  intent_name=exact_intent["name"],
                                  entities=exact_intent["entities"])

        for intent, clf in self.intent_clfs.items():
            prob = clf.predict([query.split()])[0]
            yield IntentMatch(confidence=prob,
                              intent_name=intent,
                              entities={})

    def calc_intent(self, query):
        intents = list(self.calc_intents(query))
        if len(intents):
            return max(intents, key=lambda k: k.confidence)
        return None

    def train(self):
        datasets = {intent: self.get_dataset(intent) for intent in self.intent_lines}
        for intent, (X2, y2) in datasets.items():
            if any("{" in s for s in self.intent_lines[intent]):
                continue  # regex intent
            nintent = EnpadadousSklearn()
            nintent.train(X2, y2)
            self.intent_clfs[intent] = nintent
        self.padaos.compile()

    def get_dataset(self, intent_name):
        assert intent_name in self.intent_lines
        X = []
        y = []
        for intent, samples in self.intent_lines.items():
            for s in self.intent_lines[intent_name]:
                X.append(s)
                y.append(intent)
        X2, y2 = EnpadadousSklearn.intent2dataset(X, y, intent_name, self.entity_lines)
        return X2, y2

    def _is_exact(self, intent, sample):
        exact = self.padaos.calc_intent(sample)
        if exact["name"] is not None:
            if exact["name"] == intent:
                return True
        return False

    def stats(self, intent_samples, thresh=0.5):
        intent_scores = {}

        def _get_prob(i, s):
            if self._is_exact(i, s):
                prob = 1.0
            elif clf is not None:  # regex intents dont have a clf
                prob = clf.predict([s.split()])[0]
            else:
                prob = 0.0
            return prob

        for intent, samples in intent_samples.items():
            clf = self.intent_clfs.get(intent)
            tp = 0
            fp = 0
            tn = 0
            fn = 0

            for s in list(samples):
                if "{" in s:
                    for ent, entsamples in self.entity_lines.items():
                        k = "{" + ent + "}"
                        if k in s:
                            samples += [s.replace(k, es) for es in entsamples]
                    continue

            for s in samples:
                if "{" in s:
                    continue
                prob = _get_prob(intent, s)
                if prob >= thresh:
                    tp += 1
                else:
                    fn += 1

            for intent2, samples2 in intent_samples.items():
                if intent2 == intent:
                    continue

                for s in list(samples2):
                    if "{" in s:
                        for ent, entsamples in self.entity_lines.items():
                            k = "{" + ent + "}"
                            if k in s:
                                samples2 += [s.replace(k, es) for es in entsamples]
                        continue

                for s in samples2:
                    if "{" in s:
                        continue
                    prob = _get_prob(intent, s)
                    if prob >= thresh:
                        fp += 1
                    else:
                        tn += 1

            acc = (tp + tn) / (tp + tn + fn + fp)
            #  print(intent, "accuracy:", acc,
            #        "true positives:", tp,
            #        "true negatives:", tn,
            #        "false positives:", fp,
            #        "false negatives:", fn)
            intent_scores[intent] = {"accuracy": acc,
                                     "fp": fp, "fn": fn,
                                     "tp": tp, "tn": tn}
        return intent_scores


if __name__ == "__main__":

    hello = ["hello human", "hello there", "hey", "hello", "hi"]
    name = ["my name is {name}", "call me {name}", "I am {name}",
            "the name is {name}", "{name} is my name", "{name} is my name"]
    joke = ["tell me a joke", "say a joke", "tell joke"]

    # single clf
    clf = SVC(probability=True)
    # multiple classifiers will use soft voting to select prediction
    clf = [SVC(probability=True), LogisticRegression(), Perceptron()]

    engine = EnpadadousIntentContainer(clf)

    engine.add_entity("name", ["jarbas", "bob"])
    engine.add_intent("hello", hello)
    engine.add_intent("name", name)
    engine.add_intent("joke", joke)

    engine.train()

    test_set = {"name": ["I am groot", "my name is jarbas", "jarbas is the name"],
                "hello": ["hello beautiful", "hello bob", "hello world"],
                "joke": ["say a joke", "make me laugh", "do you know any joke"]}

    print(engine.stats(test_set))
    # {'name': {'accuracy': 0.7777777777777778, 'fp': 0, 'fn': 2, 'tp': 1, 'tn': 6},
    # 'hello': {'accuracy': 0.6666666666666666, 'fp': 0, 'fn': 3, 'tp': 0, 'tn': 6},
    # 'joke': {'accuracy': 0.7777777777777778, 'fp': 0, 'fn': 2, 'tp': 1, 'tn': 6}}

    for intent, sents in test_set.items():
        for sent in sents:
            print(sent, engine.calc_intent(sent))
    # I am groot IntentMatch(intent_name='hello', confidence=0.3514303673424219, entities={})
    # my name is jarbas IntentMatch(intent_name='name', confidence=1.0, entities={'name': 'jarbas'})
    # jarbas is the name IntentMatch(intent_name='hello', confidence=0.37051078293430467, entities={})
    # hello beautiful IntentMatch(intent_name='hello', confidence=0.4355622565935733, entities={})
    # hello bob IntentMatch(intent_name='hello', confidence=0.4355622565935733, entities={})
    # hello world IntentMatch(intent_name='hello', confidence=0.4355622565935733, entities={})
    # say a joke IntentMatch(intent_name='joke', confidence=1.0, entities={})
    # make me laugh IntentMatch(intent_name='hello', confidence=0.3514303673424219, entities={})
    # do you know any joke IntentMatch(intent_name='hello', confidence=0.3830993309130246, entities={})
