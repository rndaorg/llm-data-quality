from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    "Machine learning is useful",
    "Machine learning is very useful",
    "Neural networks are powerful"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

sim = cosine_similarity(X)

threshold = 0.80
keep = []

for i in range(len(docs)):
    duplicate = False

    for j in keep:
        if sim[i][j] > threshold:
            duplicate = True
            break

    if not duplicate:
        keep.append(i)

filtered_docs = [docs[i] for i in keep]

print(filtered_docs)
