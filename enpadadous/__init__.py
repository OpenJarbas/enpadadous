from enpadadous.padaos_engine import PadaosIntentContainer
from dataclasses import dataclass
from enpadadous.clf import PadatiousSklearn


@dataclass()
class IntentMatch:
    intent_name: str
    confidence: float
    entities: dict


class EnpadadousIntentContainer:
    def __init__(self):
        self.padaos = PadaosIntentContainer()
        self.padatious = PadatiousSklearn()
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
            prob = clf.predict([query.split()])
            yield IntentMatch(confidence=prob,
                              intent_name=intent,
                              entities={})

    def calc_intent(self, query):
        intents = list(self.calc_intents(query))
        if len(intents):
            return max(intents, key=lambda k: k.confidence)
        return None

    def train(self):
        X = []
        y = []
        for intent, samples in self.intent_lines.items():
            for s in samples:
                X.append(s)
                y.append(intent)

        for intent in self.intent_lines:
            X2, y2 = PadatiousSklearn.intent2dataset(X, y, intent)
            nintent = PadatiousSklearn()
            nintent.train(X2, y2)
            self.intent_clfs[intent] = nintent
        self.padaos.compile()


if __name__ == "__main__":

    hello = ["hello world", "hello there", "hey", "hello", "hi"]
    name = ["my name is {name}", "call me {name}", "I am {name}"]
    #  name = ["my name is Bob", "call me Joe", "I am groot"]
    who = ["who are you", "how is it going", "you good"]
    joke = ["tell me a joke", "say a joke", "tell joke"]

    test = ["my name is jarbas", "jarbas is the name", "hello bob", "who are you"]

    engine = EnpadadousIntentContainer()
    engine.add_intent("hello", hello)
    engine.add_intent("name", name)
    engine.add_intent("joke", joke)
    engine.add_intent("who", who)

    engine.train()

    for sent in test:
        print(sent, engine.calc_intent(sent))
    # my name is jarbas IntentMatch(intent_name='name', confidence=1.0, entities={'name': 'jarbas'})
    # jarbas is the name IntentMatch(intent_name='who', confidence=array([0.66693695]), entities={})
    # hello bob IntentMatch(intent_name='hello', confidence=array([0.75165751]), entities={})
    # who are you IntentMatch(intent_name='who', confidence=1.0, entities={})
