# Enpadadous

wip - using the padatious features with scikit-learn

features will not be optimized in this repo, just trying to get an equivalent to padatious without libfann


## Usage

```python
from enpadadous import EnpadadousIntentContainer


hello = ["hello human", "hello there", "hey", "hello", "hi"]
name = ["my name is {name}", "call me {name}", "I am {name}",
        "the name is {name}", "{name} is my name", "{name} is my name"]
joke = ["tell me a joke", "say a joke", "tell joke"]

# single clf
# clf = SVC(probability=True)

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

```