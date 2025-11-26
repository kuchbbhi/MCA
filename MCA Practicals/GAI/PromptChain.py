# Step 1: Install dependencies (run once)
# !pip install -q langchain langchain_community transformers accelerate sentencepiece torch

# Step 2: Import libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# Step 3: Load open LLaMA-like model
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

# Step 4: Create text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=60,   # shorter generation per step
    temperature=0.6,
    top_p=0.9,
    do_sample=True
)

# Step 5: Wrap it with LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Step 6: Define simpler, cleaner prompts
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize this paragraph in one sentence:\n{text}\n###"
)

question_prompt = PromptTemplate(
    input_variables=["summary"],
    template="Based on this summary, write a simple question:\n{summary}\n###"
)

answer_prompt = PromptTemplate(
    input_variables=["question"],
    template="Give a short answer to this question:\n{question}\n###"
)

# Step 7: Create LLM chains
summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")
question_chain = LLMChain(llm=llm, prompt=question_prompt, output_key="question")
answer_chain = LLMChain(llm=llm, prompt=answer_prompt, output_key="answer")

# Step 8: Combine chains
chain = SequentialChain(
    chains=[summary_chain, question_chain, answer_chain],
    input_variables=["text"],
    output_variables=["summary", "question", "answer"])

# Step 9: Input text
text = """Artificial Intelligence is transforming industries by automating tasks,
enhancing decision-making, and creating new opportunities for innovation."""

# Step 10: Run the chain
result = chain({"text": text})

# Step 11: Display clean outputs
print( "Summary:", result["summary"].strip())
print("Question:", result["question"].strip())
print(" Answer:", result["answer"].strip())