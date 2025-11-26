from google import genai
client = genai.Client()
response = client.models.generate_content(
    model="gemini-2.0-flash-lite",
    contents='''Explain Artificial Intteligence.''')
print("Gemini:", response.text)