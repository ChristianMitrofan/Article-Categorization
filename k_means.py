from __future__ import division
import pandas as pd
import csv
import re, math
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from scipy import spatial
from random import randint

#from nltk.stem.porter import *
#from nltk.corpus import stopwords

def find_position(distlist):
	pos=0
	i=0
	max_value=distlist[0]
	for dist in distlist:
		if dist > max_value:
			max_value = dist
			pos=i
		i=i+1
	return pos

def create_clusters(distances):
	clusters={}					#dictionary of the clusters(keys=0,1,2,3,4) and (values=number of article)
	cluster0=[]
	cluster1=[]
	cluster2=[]
	cluster3=[]
	cluster4=[]
	for dist in distances:
		cluster_num=find_position(distances.get(dist))
		if cluster_num == 0:
			cluster0.append(dist)
		elif cluster_num == 1:
			cluster1.append(dist)
		elif cluster_num == 2:
			cluster2.append(dist)
		elif cluster_num == 3:
			cluster3.append(dist)
		elif cluster_num == 4:
			cluster4.append(dist)
	clusters[0]=cluster0
	clusters[1]=cluster1
	clusters[2]=cluster2
	clusters[3]=cluster3
	clusters[4]=cluster4
	return clusters

def find_centroids(clusters,distances):
	centroids=[]
	for cluster in clusters:
		new_centroid=[0]*len(distances[0])
		for article in clusters.get(cluster):
			new_centroid=[x+y for x,y in zip(new_centroid,distances[article])]
		centroids.append([x/len(clusters.get(cluster)) for x in new_centroid])
	return centroids

def csv_writer(data, path):
    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter="\t")
        for line in data:
            writer.writerow(line)


def same_centroids(old_cents,new_cents):
	for i in range(0,5):
		for j in range(0,len(old_cents[0])):
			if(old_cents[i][j]!=new_cents[i][j]):
				return False
	return True	

vectorizer=CountVectorizer(stop_words='english')
df=pd.read_csv("train_set.csv",sep="\t")
'''
neutral_words=set(stopwords.words('english'))
stemmer = PorterStemmer()
for index,row in df.iterrows():
        string=""
        for word in row['Content'].split():
            if word.isalpha() and word not in neutral_words:        #Stemming and removal of frequent words
                string=string+" "+stemmer.stem(word)
        df.loc[index,'Content']=string
'''
vec = vectorizer.fit_transform(df['Content'])
le=LabelEncoder()
le.fit(df['Category'])
categs=le.transform(df["Category"])
svd=TruncatedSVD(n_components=200, random_state=42)
dimensions=svd.fit_transform(vec)
simlist={}
new_centroids=[]
old_centroids=[]
max_iterations=100
for i in range(0,5):
	new_centroids.append(dimensions[randint(1,len(df.Title))])
for i in range(0,max_iterations):
	simlist={}
	for j in range(0,len(dimensions)):		#K-means begins here
		dlist=[]
		for cent in new_centroids:
			dlist.append(1-spatial.distance.cosine(dimensions[j],cent))
		simlist[j]=dlist
	cl=create_clusters(simlist)
	old_centroids=list(new_centroids)
	new_centroids=find_centroids(cl,dimensions)
	if(same_centroids(old_centroids,new_centroids)==True):
		print "After ",i," iterations,the clusters are stable!"
		break

	

categories=["Business","Film","Football","Politics","Technology"]

rates={}		
for cluster in cl.keys():
	sums=[0,0,0,0,0]
	for index in cl.get(cluster):
		category=categs[index]
		sums[category]=sums[category]+1
	rates[cluster]=[x / len(cl.get(cluster)) for x in sums ]

rows = ["Clusters,Politics,Business,Football,Film,Technology".split(",")]
for cluster in rates.keys():
    row=[]
    row.append("Cluster"+str(cluster))
    for rate in rates.get(cluster):
        row.append(str(rate))
    rows.append(row)
path = "clustering_KMeans.csv"

csv_writer(rows, path)

