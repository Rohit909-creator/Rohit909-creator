from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances

corpus = [
    'All my cats are cute','All my dogs are cute',
    'Cats are thief sometimes','Dogs are police sometimes']

vectorizer = CountVectorizer()
features = vectorizer.fit_transform(corpus).todense()
print( vectorizer.vocabulary_)

for i in features:
    print(euclidean_distances(features[0],i))
