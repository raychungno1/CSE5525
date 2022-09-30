import os
from collections import Counter
import matplotlib.pylab as plt

from tokenizer import tokenize

filename = os.path.join(os.path.dirname(__file__), 'nyt.txt')

counter = Counter()
for line in open(filename, "r"):
    counter.update(tokenize(line))

print(counter.most_common(100))

x = ["{:.2f}".format(1.0 / rank) for rank in range(1, 101)]
y = [count for _, count in counter.most_common(100)]
plt.plot(x, y)
plt.xticks(x, rotation=90)
plt.xlabel("Inverse Rank")
plt.ylabel("Word Count")
plt.savefig("zipfs_law.png", bbox_inches="tight")
plt.show()
