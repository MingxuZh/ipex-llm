
from huggingface_hub import InferenceClient
def tgirequest():
    client = InferenceClient(model="http://10.23.185.60:8080")
    client.text_generation(prompt="Write a code for snake game")
    for token in client.text_generation("How do you make cheese?", max_new_tokens=12, stream=True):
        print(token)


if __name__ == '__main__':

    print("a")