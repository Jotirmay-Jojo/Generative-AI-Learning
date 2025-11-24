from langchain_text_splitters import RecursiveCharacterTextSplitter

text= """sumary_line

Keyword arguments:
argument -- description
Return: return_description
"""

#initializing the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=5
)

#perform the spilt
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)