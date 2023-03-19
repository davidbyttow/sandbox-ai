import tempfile
import queue
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf


class Recorder(object):
    def __init__(self, device_id=None):
        self._device_id = device_id
        self._frames = None
        self._stop_event = None
        self._finalized_event = None
        self._filename = None

    def record(self):
        def callback(indata, frames, time, status):
            if status:
                print(status)
            self._frames.put(indata.copy())

        self._stop_event = threading.Event()
        self._frames = queue.Queue()
        self._filename = tempfile.mktemp(prefix="whisper-", suffix=".wav")

        with sf.SoundFile(
            self._filename, mode="x", samplerate=44100, channels=1
        ) as file:
            with sd.InputStream(
                callback=callback, dtype=np.float32, channels=1, samplerate=44100
            ):
                while not self._stop_event.is_set():
                    file.write(self._frames.get())
        self._finalized_event.set()

    def start(self):
        self._finalized_event = threading.Event()
        threading.Thread(target=self.record).start()

    def stop(self):
        self._stop_event.set()
        while not self._finalized_event.is_set():
            pass
        return self._filename


def main():
    print("Press enter to record")
    input()
    r = Recorder()
    r.start()
    print("Press enter to stop")
    input()
    filename = r.stop()
    print(">", filename)


if __name__ == "__main__":
    main()
