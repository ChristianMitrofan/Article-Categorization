import pandas as pd
import matplotlib.pyplot as plt
from os import path
from wordcloud import WordCloud

# Read the whole text.
df = pd.read_csv("train_set.csv",sep='\t')
categories=["Politics","Film","Football","Business","Technology"]
for category in categories:
	text=""
	for index, row in df.iterrows():
		if row['Category']==category:
			text=text+row['Title']

	# Generate a word cloud image
	wordcloud = WordCloud().generate(text)

	# Display the generated image:
	# the matplotlib way:
	plt.imshow(wordcloud)
	plt.axis("off")

	# take relative word frequencies into account, lower max_font_size
	wordcloud = WordCloud(background_color='white',
	                          width=1200,
	                          height=1000).generate(text)
	#plt.figure()
	plt.imshow(wordcloud)
	plt.axis("off")
	#plt.show()
	plt.savefig(category+'.png')
