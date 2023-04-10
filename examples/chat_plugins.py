import os
import openai
import pathlib
import re

base_dir = pathlib.Path(__file__).parent
openai.api_key = os.getenv("OPENAI_API_KEY")


def new_message(role, content):
    return {"role": role, "content": content}


imagine_pattern = r"\{imagine[\s\S]*?\}"
plugin_replacement = "\033[94m\033[1m" + "\g<0>" + "\033[0m\033[0m"


def start_chat():
    messages = [
        new_message(
            "system",
            'You are an useful AI that has unique capabilities (aka, plugins). Here are your capabilities: The ability to create images from a prompt. When you need to generate an image or are asked to do so, you can simply respond with the following: {imagine "prompt"}. The prompt can be up to 512 characters. For example: {imagine "a fat yellow dog"}',
        )
    ]

    print("\n")

    while True:
        prompt = input("\033[92m\033[1m" + "? " + "\033[0m\033[0m")
        if prompt == "quit":
            break
        messages.append(new_message("user", prompt))

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages, max_tokens=200
        )

        msg = response["choices"][0]["message"]
        resp = msg["content"]
        resp = re.sub(imagine_pattern, plugin_replacement, resp)
        print("\033[93m\033[1m" + f"\nAI:" + "\033[0m\033[0m", resp)
        messages.append(msg)


if __name__ == "__main__":
    start_chat()
