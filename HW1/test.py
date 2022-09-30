from models import *

i = Indexer()
ext = BetterFeatureExtractor(i)

sentences = ["The car is driven on the road", "The truck is driven on the highway"]
sentences = [s.split(" ") for s in sentences]
print(sentences)
for s in sentences:
    ext.extract_features(s, True)

ext.calculate_idf()

for s in sentences:
    ext.extract_features(s)