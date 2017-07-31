# this API is mostly written by Noa Naaman.
# feature engineering is done by myself (Justin Su)

import re
from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


class Mention(object):
    def __init__(self,tokens,sentenceID,span,label):
        self.tokens = tokens
        self.sentenceID = sentenceID
        self.span = span
        self.label = label

    def features(self):
        f = dict()

        f['text'] = ' '.join([t.annotations['Word itself'] for t in self.tokens]).lower()
        #f['NE'] = self.tokens[0].annotations['Named Entities'].strip('(*)')
        #f['nested_tree'] = re.sub(r'.*\((\w+)\*+\).*', r'\1', ''.join([t.annotations['Parse bit'] for t in self.tokens]))
        f['definite'] = 1 if self.tokens[0].text.lower() == 'the' else 0
        f['demonstrative'] = 1 if self.tokens[0].text.lower() in {'this', 'that', 'these', 'those'} else 0
        f['POS'] = ' '.join([t.annotations['POS'] for t in self.tokens])
        f['speaker'] = self.tokens[0].annotations['Speaker/Author']
        w = self.tokens[0].annotations['Word itself'].lower()
        if w in ['i', 'me', 'you', 'my', 'your', 'we', 'us', 'our']:
            f['pronoun'] = 1
        elif w in ['he', 'him', 'his']:
            f['pronoun'] = 2
        elif w in ['she', 'her']:
            f['pronoun'] = 3
        elif w in ['it', 'its', 'they', 'their', 'them']:
            f['pronoun'] = 4
        else:
            f['pronoun'] = 0
        return f

    def write_results(self, clusterID):
        self.tokens[0].predicted_coref['start'].add(clusterID)
        self.tokens[-1].predicted_coref['end'].add(clusterID)


class MentionPair(object):

    def __init__(self,antecedent,anaphor):
        self.antecedent = antecedent
        self.anaphor = anaphor
        if antecedent.label==anaphor.label:
            self.label = True
        else:
            self.label = False

    def features(self):
        antecedent_features = self.antecedent.features()
        anaphor_features = self.anaphor.features()

        f = {'antecedent_'+x:antecedent_features[x] for x in antecedent_features if x not in ['text', 'POS', 'speaker'] and antecedent_features[x] != None}
        for x in anaphor_features:
            if anaphor_features[x]!= None and x not in ['text', 'POS', 'speaker']:
                f['anaphor_'+x] = anaphor_features[x]

        f['same_tokens'] = 1 if antecedent_features['text'] == anaphor_features['text'] else 0
        f['same_NE'] = 1 if antecedent_features['NE'] == anaphor_features['NE'] else 0
        f['distance'] = self.anaphor.sentenceID - self.antecedent.sentenceID
        f['same_constituent'] = 1 if antecedent_features['nested_tree'] == anaphor_features['nested_tree'] else 0
        f['same_POS'] = 1 if antecedent_features['POS'] == anaphor_features['POS'] else 0
        f['partial_token_match'] = 1 if similar(antecedent_features['text'], anaphor_features['text']) > 0.5 else 0
        f['partial_POS_match'] = 1 if similar(antecedent_features['text'], anaphor_features['text']) > 0.5 else 0
        f['same_speaker'] = 1 if anaphor_features['speaker'] == antecedent_features['speaker'] else 0

        return f
