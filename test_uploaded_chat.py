# test_uploaded_chat.py
from tools.upload_and_ingest import ingest_file
from tools.chat_with_uploaded_docs import ask_question_from_uploaded_doc

user_id = "user_001"

# Upload file
file_path = "./docs/sample1.txt"
ingest_file(file_path, user_id=user_id)

# Ask question
query = "What is this document about?"
answer = ask_question_from_uploaded_doc(query, user_id=user_id)
print("🤖", answer)
