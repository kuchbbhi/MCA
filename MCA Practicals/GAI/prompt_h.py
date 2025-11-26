from transformers import pipeline
generator = pipeline('text-generation', model='gpt2')
prompt = "Explain Artificial Intelligence."
outputs = generator(prompt, max_length=50, num_return_sequences=1)
for i, output in enumerate(outputs):
    print(f"Generated Text {i+1}: {output['generated_text']}")