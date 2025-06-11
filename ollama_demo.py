from ollama import chat

stream = chat(
    model='llama3.1',
    messages=[{'role': 'user', 'content': '三角函数'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
