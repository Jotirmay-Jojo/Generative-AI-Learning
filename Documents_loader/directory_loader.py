from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader = DirectoryLoader(
    path='Documents_loader',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs=loader.lazy_load()
print(docs[0].page_content)