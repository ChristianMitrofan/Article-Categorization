import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn import metrics,preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import label_binarize
from nltk.stem.porter import *
from nltk.corpus import stopwords

#Classifier imports
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.cross_validation import KFold

def csv_writer(data, path):
    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter="\t")
        for line in data:
            writer.writerow(line)

#Read Data
df=pd.read_csv("train_set.csv",sep="\t")
le = preprocessing.LabelEncoder()
le.fit(df["Category"])
Y_train=le.transform(df["Category"])
X_train=df['Content']
vectorizer=CountVectorizer(stop_words='english')
transformer=TfidfTransformer()
svd=TruncatedSVD(random_state=22)
svc=SVC(probability=True)
mnb=MultinomialNB()
bnb=BernoulliNB()
knn=KNeighborsClassifier()
rfc=RandomForestClassifier()
vcl = VotingClassifier(estimators=[('svc', svc), ('knn', knn), ('rfc', rfc)], voting='soft')

neutral_words=set(stopwords.words('english'))
stemmer = PorterStemmer()

#classifier_type = raw_input("Which classifier to use? (1.SVC 2.MNB 3.BNB 4.KNN 5.RFC 6.OUR)")
#print "You chose %s." % classifier_type
accuracies=["Accuracy"]
rocs=["ROC"]
best_accuracy=0
clf=1
for classifier_type in range(1,7):
    if(classifier_type==1):
        pipeline = Pipeline([
            ('vect', vectorizer),
            ('tfidf', transformer),
            ('svd', svd),
            ('svc', svc),])
    elif(classifier_type==2):
        pipeline = Pipeline([
            ('vect', vectorizer),
            ('tfidf', transformer),
            ('mnb', mnb),])
    elif(classifier_type==3):
        pipeline = Pipeline([
            ('vect', vectorizer),
            ('tfidf', transformer),
            ('svd', svd),
            ('bnb', bnb),]) 
    elif(classifier_type==4):
        pipeline = Pipeline([
            ('vect', vectorizer),
            ('tfidf', transformer),
            ('svd', svd),
            ('knn', knn),])     
    elif(classifier_type==5):
        pipeline = Pipeline([
            ('vect', vectorizer),
            ('tfidf', transformer),
            ('svd', svd),
            ('rfc', rfc),])
    elif(classifier_type==6):       #We use votingClassifier which consists of SVC,KNeighborsClassifier and RandomForestClassifier
        pipeline = Pipeline([       #and we stem every word of all the contents and remove the most frequent ones
            ('vect', vectorizer),
            ('tfidf', transformer),
            ('svd', svd),
            ('vcl', vcl),])
        for index,row in df.iterrows():
            string=""
            for word in row['Content'].split():
                if word.isalpha() and word not in neutral_words:        #Stemming and removal of frequent words
                    string=string+" "+stemmer.stem(word)
            df.loc[index,'Content']=string


    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []                                            
    acsum=0
    kf=KFold(n=len(df.Title), n_folds=10, shuffle=False,random_state=None)
    for i, (train, test) in enumerate(kf):
        pipeline.fit(X_train[train], Y_train[train])
        probabilities = pipeline.predict_proba(X_train[test])
        predictions = pipeline.predict(X_train[test])
        score=metrics.accuracy_score(Y_train[test], predictions)
        acsum=acsum+score
        for i in range(0,5):
            fpr, tpr, thresholds = roc_curve(Y_train[test], probabilities[:,i], pos_label=0)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        #plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    meanac=acsum/10
    if(meanac>best_accuracy):
        clf=classifier_type
    accuracies.append(str(meanac))

    #plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    mean_tpr /= len(kf)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    rocs.append(str(mean_auc))
    if(classifier_type==1):
        plt.plot(mean_fpr, mean_tpr, 'k--',label='Svc mean ROC = %0.2f' % mean_auc, lw=2, color='r')
    elif(classifier_type==2):
        plt.plot(mean_fpr, mean_tpr, 'k--',label='Mnb mean ROC = %0.2f' % mean_auc, lw=2, color='b')
    elif(classifier_type==3):
        plt.plot(mean_fpr, mean_tpr, 'k--',label='Bnb mean ROC = %0.2f' % mean_auc, lw=2, color='g')
    elif(classifier_type==4):
        plt.plot(mean_fpr, mean_tpr, 'k--',label='Knn mean ROC = %0.2f' % mean_auc, lw=2, color='c')
    elif(classifier_type==5):
        plt.plot(mean_fpr, mean_tpr, 'k--',label='Rfc mean ROC = %0.2f' % mean_auc, lw=2, color='m')
    elif(classifier_type==6):
        plt.plot(mean_fpr, mean_tpr, 'k--',label='Our mean ROC = %0.2f' % mean_auc, lw=2, color='k')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="upper left")

plt.savefig("roc_10fold"+".png")

rows = ["Statistic Measure,SVM,Naive Bayes(MNB),Naive Bayes(BNB),KNN,Random Forest,My Method".split(",")]
rows.append(accuracies)
rows.append(rocs)
path = "EvaluationMetric_10fold.csv"

csv_writer(rows, path)

df_test=pd.read_csv("test_set.csv",sep="\t")
X_test=df_test['Content']

if(classifier_type==1):
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('tfidf', transformer),
        ('svd', svd),
        ('svc', svc),])
elif(classifier_type==2):
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('tfidf', transformer),
        ('mnb', mnb),])
elif(classifier_type==3):
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('tfidf', transformer),
        ('svd', svd),
        ('bnb', bnb),]) 
elif(classifier_type==4):
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('tfidf', transformer),
        ('svd', svd),
        ('knn', knn),])     
elif(classifier_type==5):
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('tfidf', transformer),
        ('svd', svd),
        ('rfc', rfc),])
elif(classifier_type==6):
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('tfidf', transformer),
        ('svd', svd),
        ('vcl', vcl),])
    '''
    for index,row in df_test.iterrows():
        string=""
        for word in row['Content'].split():
            if word.isalpha() and word not in neutral_words:        #Stemming and removal of frequent words
                string=string+" "+stemmer.stem(word)
        df_test.loc[index,'Content']=string
    '''
pipeline.fit(X_train, Y_train)
predictions = pipeline.predict(X_test)
pr=le.inverse_transform(predictions)
rows = ["Id,Predicted Category".split(",")]
for index, row in df_test.iterrows():
    r=[]
    r.append(str(row['Id']))
    r.append(pr[index])
    rows.append(r)

path = "testSet_categories.csv"

csv_writer(rows, path)

