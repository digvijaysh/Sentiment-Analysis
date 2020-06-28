import numpy as np
import scipy as sp
import pandas as pd
from collections import defaultdict
import nltk


# In[118]:


stopwords = set({'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'} )


# In[119]:


with open("train.dat", "r") as fh:
    lines = fh.readlines()

y = np.loadtxt('program2/train.labels')


# In[120]:


tokenizer = nltk.tokenize.TreebankWordTokenizer()
docs = [tokenizer.tokenize(l) for l in lines]


# In[121]:


stemmer = nltk.stem.PorterStemmer()
docs1 = [ [stemmer.stem(t) for t in d] for d in docs ]


# In[122]:


docs2 = [ [t for t in d if len(t) >= 4 and len(t)<=15] for d in docs1 ]


# In[ ]:





# In[123]:


docs3 = [ [t for t in d if t not in stopwords ] for d in docs2 ]


# In[124]:


wordFreq = {}
for l in docs3:
    for d in l:
        if d in wordFreq:
            wordFreq[d] += 1
        else:
            wordFreq[d] = 1


# In[125]:


sortedWordFreq = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)


# In[ ]:





# In[126]:


wordsList = sortedWordFreq[:2000]


# In[127]:


wordsSet = set()
for i in range(0,len(wordsList)):
    wordsSet.add(wordsList[i][0])


# In[128]:


docs4 = [' '.join([stemmer.stem(word) for word in d ]) for d in docs]


# In[129]:


wordsA =[]
for line in docs4:
    a = dict.fromkeys(wordsSet,0)
    for word in line.split():
        if word in wordsSet:
            a[word] +=1 
    wordsA.append(a)


# In[130]:


frame = pd.DataFrame(wordsA)


# In[ ]:





# In[131]:


def computeTF(wordsA,line):
    tfDict = {}
    bcount = len(line)
    for word,count in wordsA.items():
        tfDict[word] = count/float(bcount)
    return tfDict


# In[132]:


tfDict = []
for i in range(0,len(wordsA)):
    t = computeTF(wordsA[i],lines[i].split())
    tfDict.append(t)


# In[133]:


def computeIDF(wordsA):
    import math
    idfDict = {}
    N = len(wordsA)
    
    idfDict = dict.fromkeys(wordsA[0].keys(),0)
    for doc in wordsA:
        for word,val in doc.items():
            if val > 0:
                idfDict[word] += 1
                
                           
    for w, v in idfDict.items():
        if v==0:
            continue
        idfDict[w] = math.log(N / float(v))
        
    return idfDict


# In[134]:


idfs = computeIDF(wordsA)


# In[135]:


def computeTFIDF(tf,idfs):
    tfidf = {}
    for word,val in tf.items():
        tfidf[word] = val*idfs[word]
    return tfidf


# In[136]:


t = []
for w in tfDict:
    t1 = computeTFIDF(w,idfs)
    t.append(t1)


# In[137]:


frame = pd.DataFrame(t)


# In[ ]:





# In[138]:


x = frame.values


# In[139]:


x = np.hstack([np.ones((x.shape[0], 1)), x])


# In[140]:


x.shape


# In[141]:


theta = np.zeros(x.shape[1])


# In[142]:


def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[143]:


def costFunction(x,y,theta):
    z=y*x.dot(theta)
    l=np.mean(np.log(1.0+np.exp(-z)))
    return l


# In[144]:


def reg(x,y,theta,Lambda):
    m = x.shape[0]
    sum1 = float(0)
    for i in range(1,theta.shape[0]):
        sum1 += theta[i]*theta[i]
    
    reg = (Lambda/(2*m))*sum1
    return reg


# In[145]:


def regularization(x,y,theta,Lambda):
    m = len(y)
    z = y*x.dot(theta)
    s = sigmoid(z)
    x0 = (1/m)*np.sum((s-1)*y*x[:,0])
    x1 = np.zeros(x.shape[1])
    for i in range(1,x.shape[1]):
        x1[i] = (1/m)*np.sum((s-1)*y*x[:,i])+(Lambda/m)*theta[i]
    
    x1[0] = x0
    return x1


# In[146]:


def cost1(x,y,theta,Lambda):
    e = costFunction(x,y,theta)
    reg1 = e + reg(x,y,theta,Lambda)
    return reg1


# In[147]:


cost_history = []
theta_history = []
iter_num = []


# In[150]:


def logistic(x,y,theta,alpha,Lambda):
    m = len(y)
    n = 0
    prevCost = cost1(x,y,theta,Lambda)
    theta_history.append(theta)
    while True:
        if n>25000:
            break;
        x1 = regularization(x,y,theta,Lambda)
        theta = theta - (alpha*x1)
        theta_history.append(theta)
        newCost = cost1(x,y,theta,Lambda)
        cost_history.append(prevCost)
        prevCost = newCost
        iter_num.append(n)
        n += 1
        
        
    return theta


# In[ ]:


x_train = x[:17500]
x_val = x[17500:x.shape[0]]
y_train = y[:17500]
y_val = y[17500:x.shape[0]]
theta_train = np.zeros(x.shape[1])
theta_train = logistic(x_train,y_train,theta_train,1,0)


# In[74]:


#plt.plot(iter_num,cost_history)





# In[80]:


with open("test.dat", "r") as fh:
    lines1 = fh.readlines()


# In[81]:


tokenizer = nltk.tokenize.TreebankWordTokenizer()
docs12 = [tokenizer.tokenize(l) for l in lines1]


# In[82]:


docs43 = [' '.join([stemmer.stem(word) for word in d ]) for d in docs12]


# In[83]:


wordstest =[]
for line in docs43:
    b = dict.fromkeys(wordsSet,0)
    for word in line.split():
        if word in wordsSet:
            b[word] +=1 
    wordstest.append(b)


# In[84]:


#len(wordstest[0])


# In[85]:


frame1 = pd.DataFrame(wordstest)


# In[86]:


tfDict1 = []
for i in range(0,len(wordstest)):
    t1 = computeTF(wordstest[i],lines[i].split())
    tfDict1.append(t1)


# In[87]:


idfs1 = computeIDF(wordstest)


# In[88]:


t2 = []
for w in tfDict1:
    t3 = computeTFIDF(w,idfs1)
    t2.append(t3)


# In[89]:


frame4 = pd.DataFrame(t2)


# In[90]:


x_test = frame4.values


# In[91]:


x_test = np.hstack([np.ones((x_test.shape[0], 1)), x_test])


# In[92]:


x9 = x_test.dot(theta_train)
x9 = np.sign(x9)
x9 = x9.astype(int)


# In[93]:


np.savetxt("pr4.txt",x9,fmt='%+i')


# In[ ]:





# In[ ]:





# In[ ]:




