# Hugging Face local pipeline
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load a local Hugging Face model
generator = pipeline("text-generation", model="distilgpt2", max_new_tokens=100)
# Wrap with LangChain
llm = HuggingFacePipeline(pipeline=generator)
# Prompt template
template = """You are a helpful assistant.
Question: {question}
Answer:"""

prompt = PromptTemplate(input_variables=["question"], template=template)
# Create chain
chain = LLMChain(llm=llm, prompt=prompt)
# Run query
response = chain.run("Explain residual connections in transformers.")
print(response)