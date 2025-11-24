from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

text= """
def is_palindrome_loop(text):
    left = 0
    right = len(text) - 1
    while left < right:
        if text[left] != text[right]:
            return False
        left += 1
        right -= 1
    return True

# Example usage
word1 = "racecar"
word2 = "python"
print(f"'{word1}' is a palindrome: {is_palindrome_loop(word1)}")
print(f"'{word2}' is a palindrome: {is_palindrome_loop(word2)}")
"""
chunk = splitter.split_text(text)

print(chunk[0])