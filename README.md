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

engine = EnpadadousIntentContainer()
engine.add_entity("name", ["jarbas", "bob"])
engine.add_intent("hello", hello)
engine.add_intent("name", name)
engine.add_intent("joke", joke)

engine.train()

test = ["my name is jarbas", "jarbas is the name", "hello bob", "hello world", "do you know any joke"]

for sent in test:
    print(sent, engine.calc_intent(sent))
# my name is jarbas IntentMatch(intent_name='name', confidence=1.0, entities={'name': 'jarbas'})
# jarbas is the name IntentMatch(intent_name='hello', confidence=0.6938639951116328, entities={})
# hello bob IntentMatch(intent_name='name', confidence=0.11145063196059113, entities={})
# hello world IntentMatch(intent_name='hello', confidence=0.7203346727194004, entities={})
# do you know any joke IntentMatch(intent_name='joke', confidence=0.6801598511923101, entities={})

```