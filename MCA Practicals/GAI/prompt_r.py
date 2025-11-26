from openai import OpenAI
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="api key here",)
completion = client.chat.completions.create(
  model="openai/gpt-oss-20b:free",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }])
print(completion.choices[0].message.content)