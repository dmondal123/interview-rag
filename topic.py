import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df= pd.read_csv("java_questions.csv")
df['text']= df['question']
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer =WordNetLemmatizer()
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r' [^\w\s]', '', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in nltk.corpus.stopwords.words('english')]
    return ' '.join(words)

df['processed_text'] = df['text'].apply(preprocess_text) 
df.head()
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df ['processed_text']) 
print(X.shape)
n_topics = 4
lda =  LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)

n_top_words = 10
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    topic_names = [
        "Joins & Views",
        "Keys & Constraints", 
        "Indexes & Transactions",
        "Basic SQL Operations",
        "Views & Derived Tables",
        "Users & Security",
        "Databases & Storage",
        "Data Types & Tablespaces"
    ]
    print(f"Topic {topic_idx + 1} - {topic_names[topic_idx]}:")
    print(" ".join([feature_names[i] for i in topic.argsort() [:-n_top_words - 1: -1]]))
    df.loc[df.index == topic_idx, 'sub_topic'] = topic_names[topic_idx]

df.drop(['processed_text', 'text'], axis=1).to_csv('classified_java_questions.csv', index=False)
df.head()
'''
1. Joins & Views
2. Keys & Constraints
3. Indexes & Transactions
4. Basic SQL Operations
5. Views & Derived Tables 
6. Users & Security 
7. Databases & Storage 
8. Data Types & Tablespaces 
'''