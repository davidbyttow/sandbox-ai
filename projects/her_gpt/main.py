import os
import openai
import pathlib
import elevenlabs as el
from recorder import Recorder

base_dir = pathlib.Path(__file__).parent
openai.api_key = os.getenv("OPENAI_API_KEY")


def load_audio_file(name):
    return open(os.path.join(base_dir, "data", name), "rb")


def transcribe_file(file):
    transcript = openai.Audio.transcribe("whisper-1", file)
    text = transcript["text"]
    file.seek(0)
    return text


def new_message(role, content):
    return {"role": role, "content": content}


def record_prompt(rec):
    input("Press enter when you're finished speaking")
    tmp_file = rec.stop()
    audio_file = load_audio_file(tmp_file)
    prompt = transcribe_file(audio_file)
    return prompt


def start_chat():
    voices = el.get_available_voices()
    her_voice = next((v for v in voices if v.name() == "Bella"), None)
    if not her_voice:
        raise Exception("Could not find Bella voice")

    rec = Recorder()

    messages = [new_message("system", "You are a friend of mine.")]

    while True:
        text = input("Press enter and then say something or type 'quit' to exit")
        if text == "":
            rec.start()
        elif text == "quit":
            break

        prompt = record_prompt(rec)
        messages.append(new_message("user", prompt))
        print("\n> You:", prompt)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages, max_tokens=100
        )
        her_msg = response["choices"][0]["message"]
        her_resp = her_msg["content"]
        print("\nHer:", her_resp)
        her_voice.say(her_resp, background=False)
        messages.append(her_msg)


if __name__ == "__main__":
    start_chat()
