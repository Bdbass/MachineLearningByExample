from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import names
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 


# categories = ['comp.graphics', 'comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware',
#  'talk.religion.misc', 'comp.windows.x', 'comp.sys.mac.hardware']
categories = ['talk.religion.misc', 'comp.graphics', 'sci.space', 'alt.atheism']

features = 500

def is_letter(word):
	for char in word:
		if not char.isalpha():
			return False
		return True


def preprocessing(data):
	all_names = set(names.words())
	lemmatizer = WordNetLemmatizer()
	data_cleaned = []
	for doc in data:
		doc_cleaned = ' '.join(lemmatizer.lemmatize(word) 
								for word in doc.split()
									if is_letter(word) and 
									word not in all_names)
		data_cleaned.append(doc_cleaned)
	print('preprocessing complete')
	return data_cleaned


def process_docs(groups, features):
	data_cleaned = preprocessing(groups.data)
	#uses term-frequency 
	#count_vector = CountVectorizer(stop_words='english', max_features=None, max_df=0.5, min_df=2)
	#uses frequency inverse document frequency
	tfidf_vector = TfidfVectorizer(stop_words='english', max_features=features, max_df=0.5, min_df=2)
	data_cleaned_count = tfidf_vector.fit_transform(groups.data)
	print('processing complete')
	return data_cleaned_count, tfidf_vector


def downloads():
	nltk.download('names')
	nltk.download('wordnet')
	print('download complete')


def graph(data_cleaned_count, groups):
	print('t-sne reduction')
	tsne_model = TSNE(n_components=2, perplexity=40, random_state=42, learning_rate=500)
	data_tsne = tsne_model.fit_transform(data_cleaned_count.toarray())
	print('graphing')
	# ax.scatter(data_tsne[:, 0], data_tsne[:, 1], c=groups.target)
	for group in range(len(groups.target_names)):
		plt.scatter([data_tsne[i, 0] for i, x in enumerate(groups.target) if x == group],
			[data_tsne[j, 1] for j, x in enumerate(groups.target) if x == group], label=groups.target_names[group])

	plt.legend()
	plt.show()


def main():
	# downloads() #just run once
	groups = fetch_20newsgroups(categories=categories)
	data_cleaned_count = process_docs(groups, features)[0]
	graph(data_cleaned_count, groups)


#main()
