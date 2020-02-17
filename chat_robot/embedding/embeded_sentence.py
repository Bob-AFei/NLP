#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.models import word2vec
import numpy as np
import pandas


# In[31]:


class embeded_sentence:
    def __init__ (self, alpha = 1e-4, size = 100, n = -5):
        self.alpha = alpha
        self.size = size # size is the sentence vectors size
        self.n = n
        # self.sent_vec
        # self.pr
        # self.u = np.zeros(1)
        
    def fit(self, data):
        self.data = data
        self.sent_vec = np.zeros([len(self.data),self.size])
        self.get_pr()
        self.get_model()
        self.get_sentences()
        
    # get the probability
    def get_pr (self):
        words_pr = {}
        counter = 0
        for sentece in self.data:
            for word in sentece:
                if word in words_pr: words_pr[word]+=1
                else: words_pr[word] = 1
                counter += 1
        for x in words_pr:
            words_pr[x] /= counter
        self.pr = words_pr

    # get the w2v.model
    def get_model(self):
        self.model = word2vec.Word2Vec(self.data, size=self.size, hs=1, min_count=1, window=3,iter = 5,sorted_vocab = 1)
    
    # get the sentence vector
    def get_sentences (self):
        for i,sentence in enumerate (self.data):
            vs_temp = np.zeros(self.size)
            for word in sentence:
                if word in self.model.wv: vs_temp += self.model.wv[word]*self.alpha/(self.alpha+self.pr[word])
                else: vs_temp += 0
#                 vs_temp += self.model[word]*self.alpha/(self.alpha+self.pr[word])
            self.sent_vec[i] = vs_temp/np.linalg.norm(vs_temp)
        self.get_singlar_vc()
        self.sent_vec = self.sent_vec - (self.u@self.sent_vec.T).T
        
    # get the sentence vector
    def transform (self, data):
        vs = np.zeros([len(data),self.size])
        for i,sentence in enumerate (data):
            vs_temp = np.zeros(self.size)
            for word in sentence: 
                if word in self.model.wv: vs_temp += self.model.wv[word]*self.alpha/(self.alpha+self.pr[word])
                else: vs_temp += 0
#                 vs_temp += self.model[word]*self.alpha/(self.alpha+self.pr[word])
            vs[i] = vs_temp/len(sentence)
        vs = vs - (self.u@vs.T).T
        return vs
        
    # get the singlar vector
    def get_singlar_vc (self):
        x = self.sent_vec-np.average(self.sent_vec)
        cov = np.cov(x.T)
        a,v = np.linalg.eig(cov)
        u = v[self.n:,:].T
        self.u = u@u.T

    def np_cosine (self, a1, a2):
        a = np.dot(a1,a2)/(np.linalg.norm(a1)*np.linalg.norm(a2))
        return a
    
    def most_similar (self,given_s,n = 5):
        given_sv = self.transform(given_s)
        similarity = np.array([])
        for i,vs in enumerate (self.sent_vec):  # TFIDF
            similarity = np.append (similarity, self.np_cosine(given_sv,vs))
        a = np.argsort(similarity)
        return a[-n:],similarity[a[-n:]]

    def predict (self,X,n=5, threshold=0.1):
        predict = np.array([])
        similar = np.array([])
        for x in X:
            a,b = self.most_similar([x],n)
#             if b[-1] < threshold or np.max(b) - np.min(b) < 0.1 : a = -np.ones(n)
            if np.max(b) - np.min(b) < threshold : a = -np.ones(n)
            predict = np.append(predict,a)
            similar = np.append(similar,b)
        return predict.reshape(len(X),n),similar.reshape(len(X),n)
    
    def score(self,X,y,threshold=0.1):
        l,s = self.predict(X,10,threshold)
        correct = 0
        n = len(X)
        for i in range(n):
            if np.any(l[i]+1 == y[i]) or (l[i,0] == -1 and y[i] == -1):
#                 print(l[i],y[i])
                correct+=1
        return correct/n


# In[32]:


def get_data (path):
    file = open(path,'r')
    data = file.readlines()
    questions = data[0::2]
    answers = data[1::2]
    for i in range(len(questions)):
        questions[i] = questions[i][:-1].lower().split(' ')
    return questions,answers


# In[33]:


def get_data_new (path):
    data = pandas.read_csv(path).values
    for i,temp in enumerate (data):
        data[i,0] = data[i,0].lower().split(' ')
    return data


# In[34]:


data = get_data_new('Q&A.csv') # the question is at 0-693
questions = data[:250]
answers = data[250:]


# In[35]:
ies = embeded_sentence(1e-3,100,n = -5)
ies.fit(questions[:,0])
ies.score(answers[:,0],answers[:,2])


# In[1]:


q = 'how to do enrolment'
a1,b1 = ies.predict([q.split(' ')])


# In[68]:


for i in a1.flatten():
    if i is -1: print('Error!')
    print(' '.join(questions[int(i),0]))

