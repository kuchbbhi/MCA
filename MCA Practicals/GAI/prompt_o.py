from openai import OpenAI
response = OpenAI().responses.create(
    model="gpt-5",
    input="Write a short bedtime story about a unicorn.")
print(response.output_text)