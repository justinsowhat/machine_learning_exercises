# this API is mostly written by Noa Naaman.
# I made my own modification

import os
from collections import defaultdict
from Mention import *
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC

class Token(object):
    def __init__(self,line):
        self.text = line
        self.get_annotations()
        self.predicted_coref = {'start':set(),'end':set()}

    def get_annotations(self):
        fields = self.text.split()
        self.annotations = {'Document ID': fields[0],
                            'Part number': fields[1], 'Word number': fields[2],
                            'Word itself': fields[3], 'POS': fields[4],
                            'Parse bit': fields[5], 'Predicate lemma': fields[6],
                            'Predicate Frameset ID': fields[7], 'Word sense': fields[8],
                            'Speaker/Author': fields[9], 'Named Entities':  fields[10],
                            'Predicate Arguments': [x for x in fields[11:-1]], 'Coreference': fields[-1]}
        self.coref = self.annotations['Coreference']

    def change_label(self,label):
        coref_chars = len(self.annotations['Coreference'])
        self.text = self.text[:-coref_chars-1]+label+'\n'

    def write_results(self):
        res = []
        start = ['('+str(c) for c in self.predicted_coref['start'] if c not in self.predicted_coref['end']]
        end = [str(c)+')' for c in self.predicted_coref['end'] if c not in self.predicted_coref['start']]
        complete = ['('+str(c)+')' for c in self.predicted_coref['start'].intersection(self.predicted_coref['end'])]
        parts = ['|'.join(start),'|'.join(complete),'|'.join(end)]
        s = '|'.join([part for part in parts if part])
        if s:
            self.change_label(s)
        else:
            self.change_label('-')


class Sentence(object):
    def __init__(self,tokens,sentenceID):
        self.tokens = tokens
        self.sentenceID = sentenceID
        self.mentions = []

    def collect_mentions(self):
        mention_ids = defaultdict(list)

        def get_start_ids(cr):
            return [int(x.replace(')','').replace('(','')) for x in cr.split('|') if x.startswith('(')]
        def get_end_ids(cr):
            return [int(x.replace(')','').replace('(','')) for x in cr.split('|') if x.endswith(')')]
            
        starts = [(i,t) for (i,t) in enumerate(self.tokens) if t.coref.find('(')>-1]
        starts.reverse()
        ends = [(i,t) for (i,t) in enumerate(self.tokens) if t.coref.find(')')>-1]

        for s in starts:
            ids = get_start_ids(s[1].coref)
            for i in ids:
                mention_ids[i].append(s)

        for e in ends:
            ids = get_end_ids(e[1].coref)
            for i in ids:
                s = mention_ids[i].pop()
                self.mentions.append(Mention(self.tokens[s[0]:e[0]+1],self.sentenceID,(s[0],e[0]),i))
                
    

class Document(object):
    def __init__(self,sentences):
        self.sentences = sentences

    def get_pairs(self):
        pairs = []

        N_sent = len(self.sentences)

        #collect in-sentence pairs
        for s in self.sentences:
            s.collect_mentions()
            for i in range(len(s.mentions)-1,0,-1):
                anaphor = s.mentions[i]
                for j in range(i-1,-1,-1):
                    antecedent = s.mentions[j]
                    if antecedent.span[1]<anaphor.span[0]:
                        pairs.append(MentionPair(antecedent,anaphor))
        
        #collect cross sentence pairs
        for i in range(N_sent-1,0,-1):
            for anaphor in self.sentences[i].mentions:
                for j in range(i-1,-1,-1):
                    pairs.extend([MentionPair(antecedent,anaphor) for antecedent in self.sentences[j].mentions])

        return pairs

    def set_pairs(self):
        self.pairs = self.get_pairs()        
        

    def predict(self,model,vectorizer):
        self.set_pairs()
        
        X = [p.features() for p in self.pairs]
        if len(self.pairs):
            X = vectorizer.transform(X)
            y = model.predict(X)
        else:
            y = []
        self.write_results(y)
        
    
    def cluster(self,y):
        clusterID = 0
        clusters = {}
        for i in range(len(y)):
            if y[i]==True:
                if self.pairs[i].anaphor in clusters:
                    clusters[self.pairs[i].antecedent] = clusters[self.pairs[i].anaphor]
                elif self.pairs[i].antecedent in clusters:
                    clusters[self.pairs[i].anaphor] = clusters[self.pairs[i].antecedent]
                else:
                    clusters[self.pairs[i].antecedent] = clusterID
                    clusters[self.pairs[i].anaphor] = clusterID
                    clusterID += 1
        for s in self.sentences:
            for m in s.mentions:
                if m not in clusters:
                    clusters[m] = clusterID
                    clusterID += 1

        return clusters

    def write_results(self,y):
        clusters = self.cluster(y)
        for mention in clusters:
            mention.write_results(clusters[mention])

        for s in self.sentences:
            for t in s.tokens:
                t.write_results()
                   
         
                    
class File(object):

    def __init__(self,path):
        self.path = path
        self.docs = []
        self.read()

    def read(self):
        with open(self.path,'r') as f:
            
       
            for l in f:
                if l.startswith('#begin document'):
                    sentenceID = 0
                    begin = l                    
                    doc = []
                    sent = []
                elif l == '#end document\n':
                    self.docs.append((begin,Document(doc)))
                elif l == '\n':
                    if sent:
                        doc.append(Sentence(sent,sentenceID))
                        sent = []
                        sentenceID += 1
                else:
                    sent.append(Token(l))

    def write(self):
        with open(self.path+'_out', 'w') as f:
            for d in self.docs:
                f.write(d[0])
                for sent in d[1].sentences:
                    for tok in sent.tokens:
                        f.write(tok.text)
                    f.write('\n')
                f.write('#end document\n')

    def get_features(self):
        X_y = []
        for d in self.docs:
            pairs = d[1].get_pairs()
            X_y.extend([(p.features(),p.label) for p in pairs])
        return X_y

    def predict(self,model,vectorizer):
        for d in self.docs:
            d[1].predict(model,vectorizer)
        self.write()


class Corpus(object):

    def get_files(self,path,extension):
        paths = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(extension):
                    paths.append(os.path.join(root, file))
        return paths

    def get_features(self,path,extension):
        paths = self.get_files(path,extension)
        print('got paths to files')
        print(len(paths))
        count = 0
        X_y = []
        for path in paths:
            count += 1

            if count%100 == 0:
                print(count)
            f = File(path)
            X_y.extend(f.get_features())

        return X_y
        

if __name__ == '__main__':
    dev = os.path.join('conll-2012', 'dev', 'english', 'annotations')
    train = os.path.join('conll-2012', 'train', 'english', 'annotations')
    test = os.path.join('conll-2012', 'test', 'english', 'annotations')

    vec = DictVectorizer()
    cor = Corpus()

    X_y_train = cor.get_features(train,"auto_conll")
    X_train = vec.fit_transform([x for x,y in X_y_train])
    y_train = [y for x,y in X_y_train]

    model = LogisticRegression(n_jobs=-1)
    # model =  RandomForestClassifier(n_jobs=-1)
    # model = SVC()
    model.fit(X_train, y_train)
    
    print('running it on the dev set')
    for path in cor.get_files(dev, 'auto_conll'):
        print(path)
        f = File(path)
        f.predict(model, vec)

    print('running in on the test set')
    for path in cor.get_files(test,"gold_conll"):
        print(path)
        f = File(path)
        f.predict(model,vec)

