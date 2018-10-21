#gift recommender system based on interests shown on social media

from nltk.corpus import wordnet as wn
import torch, torchtext
from sklearn.metrics.pairwise import cosine_similarity
word2vecin = torchtext.vocab.Vectors('in_pruned_lite.txt', url='http://github.com/ruyimarone/SemanticSearch/raw/master/in.zip', cache = '.')
import numpy as np

#convert all words to nouns
WN_NOUN = 'n'
WN_VERB = 'v'
WN_ADJECTIVE = 'a'
WN_ADJECTIVE_SATELLITE = 's'
WN_ADVERB = 'r'

def convert(word, from_pos, to_pos):    
    """ Transform words given from/to POS tags """
    
    synsets = wn.synsets(word, pos=from_pos)

    # Word not found
    if not synsets:
        return []

    # Get all lemmas of the word (consider 'a'and 's' equivalent)
    lemmas = []
    for s in synsets:
        for l in s.lemmas():
            if s.name().split('.')[1] == from_pos or from_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE) and s.name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE):
                lemmas += [l]

    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]

    # filter only the desired pos (consider 'a' and 's' equivalent)
    related_noun_lemmas = []

    for drf in derivationally_related_forms:
        for l in drf[1]:
            if l.synset().name().split('.')[1] == to_pos or to_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE) and l.synset().name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE):
                related_noun_lemmas += [l]

    # Extract the words from the lemmas
    words = [l.name() for l in related_noun_lemmas]
    len_words = len(words)

    # Build the result in the form of a list containing tuples (word, probability)
    result = [(w, float(words.count(w)) / len_words) for w in set(words)]
    result.sort(key=lambda w:-w[1])
    

    # return all the possibilities sorted by probability
    return result[0][0]



#input a list of interest tags and the list of attributes of a specific gift, do semantic match on interest tags and gift tags, return the raw score of similarity
def computeGiftScore(interestTags, giftTags):
    #convert all non-noun words to noun
    i = 0
    for interest in interestTags:
        type = wn.synsets(interest)[0].pos()
        if type != WN_NOUN:
            interestTags[i] = convert(interest, type, WN_NOUN)
        i += 1
    i = 0
    for gift in giftTags:
        type = wn.synsets(gift)[0].pos()
        if type != WN_NOUN:
            giftTags[i] = convert(gift, type, WN_NOUN)
        i += 1
    sum = 0
    for interest in interestTags:
        #compute average of semantic similarity
        for gift in giftTags:
            vector1 = word2vecin[interest]
            vector2 = word2vecin[gift]
            vector1s = np.reshape(vector1, (1, -1))
            vector2s = np.reshape(vector2, (1, -1))
            sum += (cosine_similarity(vector1s, vector2s))
    avg = (sum / (len(interestTags) * len(giftTags)))

    return avg

def formatData(tags):
    i = 0
    for tag in tags:
        type = wn.synsets(tag)[0].pos()
        if type != WN_NOUN:
            tags[i] = convert(tag, type, WN_NOUN)
        i += 1
    return

#dictionary key: string name of the gift value: list of attribute
gifts = {"basketball" : ['basketball', 'athletic', 'basketball', 'athlete', 'exercise', 'teenager', 'teenage', 'health', 'train'], 'photo album': ['photo', 'photography', 'anniversary', 'collection', 'family', 'friend'], 'bike' : ['travel', 'nature', 'sport', 'lifestyle', 'utility', 'transport'], 'tent' : ['nature', 'hike', 'adventure', 'activity', 'travel', 'event'], 'palette' : ['artistic', 'creation', 'personal', 'paint', 'draw', 'picture'], 'sheet music' : ['pianist', 'music', 'instrument', 'creative', 'art', 'perform']}
person1 = ['athletic', 'sport', 'basketball', 'rock', 'band', 'football', 'baseball', 'healthy', 'game', 'teenager']
#scores of each gift
scores = {}
formatData(person1)

for gift in gifts:
    tags = gifts[gift]
    formatData(tags)
    score = computeGiftScore(person1, tags) * 10
    scores[gift] = score
    #sort the dictionary
    sortedScores = [(k, scores[k]) for k in sorted(scores, key=scores.get, reverse=True)]
print(sortedScores)

    
