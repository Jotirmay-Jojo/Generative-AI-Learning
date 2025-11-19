from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
"Sachin Tendulkar: A revered former Indian batsman, he holds the record for the most runs in both Test and One Day International (ODI) cricket.",
"Virat Kohli: A prominent Indian cricketer, he is known for his consistent performance and has achieved numerous records, particularly in the limited-overs formats.",
"MS Dhoni: A former captain of the Indian cricket team, he is celebrated for his leadership and his signature finishing style in the middle of a match.",
"Deepti Sharma: An all-rounder for the Indian women's cricket team, she is recognized for her all-round contributions and has achieved the third-highest score in women's ODIs.",
"AB de Villiers: A former South African batsman, he was famous for his innovative and unconventional stroke play, earning him the nickname 'Mr. 360'"
]

query='tell me about Virat Kolhi'

doc_emb= embedding.embed_documents(documents)
quer_emb=embedding.embed_query(query)


#dono vectors 2D list hone chaiye so brackets
# print(cosine_similarity([quer_emb],doc_emb))

scores = cosine_similarity([quer_emb],doc_emb)[0]


index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print(query)
print(documents[index])
print(score)


#CPT
# ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)

# print("\Ranking:")
# for idx, score in ranked:
#     print(f"Document {idx}: score={score:.4f}->{documents[idx]}")