from langchain_community.retrievers import WikipediaRetriever

#intialize the retriver (optinal: set language and top_k)
retriver = WikipediaRetriever(top_k_results=2, lang="en")

#define your query 
query= "albert einstein"

#get relevent Wikipedia documents
docs = retriver.invoke(query)

print(len(docs))

#print retrieved content
for i, doc in enumerate(docs):
    print(f"\n--Result {i+1}--")
    print(f"Content: \n {doc.page_content}...")