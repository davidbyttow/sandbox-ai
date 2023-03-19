import io
import os
import logging
import json
import threading

import requests
import sounddevice as sd
import soundfile
import soundfile as sf


api_key = os.getenv("ELEVENLABS_API_KEY")
BASE_URL = "https://api.elevenlabs.io/v1"
DEFAULT_HEADERS = {"accept": "*/*"}


class Voice(object):
    def __init__(self, data):
        self._data = data

    def voice_id(self):
        return self._data["voice_id"]

    def name(self):
        return self._data["name"]

    def say(self, text, background=False):
        data = generate_audio(self.voice_id(), text)
        play_audio(data, background=background)

    def __str__(self) -> str:
        return json.dumps(self._data)


def generate_audio(voice_id, text):
    payload = {"text": text, "voice_settings": {"stability": 0, "similarity_boost": 0}}
    response = _post(f"/text-to-speech/{voice_id}/stream", json_data=payload)
    return response.content


def play_audio(
    audio_data,
    background=False,
    on_start=lambda: None,
    on_end=lambda: None,
    device_id=None,
):
    if device_id is None:
        device_id = sd.default.device[1]
    wrapper = _SDPlaybackWrapper(
        data=audio_data, device_id=device_id, on_start=on_start, on_end=on_end
    )
    if not background:
        with wrapper.stream:
            wrapper.end_playback_event.wait()
    else:
        wrapper.stream.start()


def get_available_voices():
    response = _get("/voices")
    data = response.json()
    return [Voice(d) for d in data["voices"]]


class _SDPlaybackWrapper:
    def __init__(
        self,
        data,
        device_id,
        on_start=lambda: None,
        on_end=lambda: None,
    ):
        sound_file = sf.SoundFile(io.BytesIO(data))
        sound_file.seek(0)
        self.on_start = on_start
        self.on_end = on_end
        self.start_playback_event = threading.Event()
        self.end_playback_event = threading.Event()
        self.data = sound_file.read(always_2d=True)
        self.currentFrame = 0
        self.stream = sd.OutputStream(
            channels=sound_file.channels,
            callback=self.callback,
            samplerate=sound_file.samplerate,
            device=device_id,
            finished_callback=self.end_playback,
        )

    def callback(self, outdata, frames, time, status):
        if status:
            print(status)

        if not self.start_playback_event.is_set():
            self.start_playback_event.set()
            self.on_start()

        chunksize = min(len(self.data) - self.currentFrame, frames)
        outdata[:chunksize] = self.data[
            self.currentFrame : self.currentFrame + chunksize
        ]
        if chunksize < frames:
            outdata[chunksize:] = 0
            raise sd.CallbackStop()
        self.currentFrame += chunksize

    def end_playback(self):
        self.on_end()
        self.end_playback_event.set()


def _get(path, headers=DEFAULT_HEADERS):
    return _call_api("GET", path, headers)


def _post(path, json_data, headers=DEFAULT_HEADERS):
    return _call_api("POST", path, headers, json_data)


def _call_api(method, path, headers=DEFAULT_HEADERS, json_data=None, file_data=None):
    if not api_key:
        raise Exception("No API key provided")
    headers["xi-api-key"] = api_key
    if path[0] != "/":
        path = "/" + path
    endpoint = BASE_URL + path
    if method == "GET":
        response = requests.get(endpoint, headers=headers)
    elif method == "POST":
        response = requests.post(endpoint, headers=headers, json=json_data)
    elif method == "DEL":
        response = requests.delete(endpoint, headers=headers)
    elif method == "MULTIPART":
        response = requests.post(
            endpoint, headers=headers, data=json_data, files=file_data
        )
    else:
        raise Exception("Unknown method: " + method)

    response.raise_for_status()
    return response
