import os
from collections import Counter

filename = os.path.join(os.path.dirname(__file__), 'nyt.txt')

counter = Counter()
for line in open(filename, "r"):
    counter.update(line.split())

print(counter.most_common(10))
