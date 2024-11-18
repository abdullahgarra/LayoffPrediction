from sentence_transformers import SentenceTransformer


model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def get_embedding(text):
    return model.encode(text)

def get_example_embeddings(example_sentences):
    return [get_embedding(sentence) for sentence in example_sentences]
