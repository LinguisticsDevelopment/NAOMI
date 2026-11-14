import json

words = [
    'The',
    'boy',
    'is',
    'fast'
]
possible = [
    ['DET'],
    ['NOUN', 'ADJ'],
    ['VERB'],
    ['VERB', 'ADJ', 'ADV']
]

with open("rules.json") as f:
    rules = json.loads(f.read())

for i in range(len(words)):
    for rule in rules:
        if len(rule['prev']) > 0 and i == 0: continue
        if len(rule['next']) > 0 and i == len(words)-1: continue

        if any(x in possible[i-1] for x in rule['prev']):
            if any(x in possible[i] for x in rule['curr']):
                # if any(x in possible[i+1] for x in rule['next']):
                print("resolving")
                possible[i] = [rule['resolve']]

print(possible)

