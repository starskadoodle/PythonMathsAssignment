import math
import numpy as np

def build_dictionary(docs):
    dictionary = set()
    for doc in docs:
        words = doc.split()
        dictionary.update(words)
    return dictionary

def build_inverted_index(docs):
    inverted_index = {}
    for doc_id, doc in enumerate(docs, start=1):
        words = doc.split()
        for word in words:
            if word not in inverted_index:
                inverted_index[word] = []
            inverted_index[word].append(doc_id)
    return inverted_index

def calculate_angle(query_vector, doc_vector):
    dot_product = np.dot(query_vector, doc_vector)
    query_norm = np.linalg.norm(query_vector)
    doc_norm = np.linalg.norm(doc_vector)
    cosine_similarity = dot_product / (query_norm * doc_norm)
    angle = math.degrees(math.acos(cosine_similarity))
    return angle

def search_documents(docs, queries):
    dictionary = build_dictionary(docs)
    inverted_index = build_inverted_index(docs)

    print("Words in dictionary:", len(dictionary))

    word_indices = {word: i for i, word in enumerate(dictionary)}

    for query in queries:
        print("Query:", query)
        query_vector = np.zeros(len(dictionary))
        query_words = query.split()
        for word in query_words:
            if word in word_indices:
                query_vector[word_indices[word]] += 1

        relevant_doc_ids = np.unique(np.concatenate([inverted_index.get(word, []) for word in query_words if word in inverted_index]))
        print("Relevant documents:", *relevant_doc_ids)

        doc_vectors = {}
        for doc_id in relevant_doc_ids:
            doc = docs[doc_id - 1]
            doc_vector = np.zeros_like(query_vector)
            for word in doc.split():
                if word in word_indices:
                    doc_vector[word_indices[word]] += 1    
            angle = calculate_angle(query_vector, doc_vector)
            doc_vectors[doc_id] = angle

        sorted_docs = sorted(doc_vectors.items(), key=lambda x: x[1], reverse=True)
        for doc_id, angle in sorted_docs:
            print(doc_id, "{:.2f}".format(angle)) 
        print()

# Read input files
with open("docs.txt", "r") as docs_file, open("queries.txt", "r") as queries_file:
    docs = docs_file.read().splitlines()
    queries = queries_file.read().splitlines()

# Perform document search
search_documents(docs, queries)