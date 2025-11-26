from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
template2 = """
You are a smart tutor.
Answer the question based on examples:

Example 1:
Q: What is 2+2?
A: 4

Example 2:
Q: What is the capital of France?
A: Paris

Now answer:
Q: {question}
"""

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite')
prompt2 = PromptTemplate(input_variables=["question"], template=template2)
chain = LLMChain(llm=llm, prompt=prompt2)

print(prompt2.format(question="What is the capital of Germany?"))
print(chain.run({"question": "What is the capital of Germany?"}))