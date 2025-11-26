# pip install langchain-google-genai
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI( model='gemini-2.0-flash-lite') # GOOGLE_API_KEY
while True:
    user_input = input("You: ")
    print(f"You: {user_input}")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break
    response = llm.predict(user_input)
    print("Chatbot:", response)