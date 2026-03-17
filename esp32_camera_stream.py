import numpy as np
import cv2
import sys
import requests
import threading

# класс для приема потока с камеры
class ESP32CameraStream:
    def __init__(self, url):
        self.url = url
        self.frame = None
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        session = requests.Session()
        try:
            response = session.get(self.url, stream=True, timeout=5)
            bytes_data = b''
            for chunk in response.iter_content(chunk_size=1024):
                if not self.running:
                    break
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8') # Начало JPEG
                b = bytes_data.find(b'\xff\xd9') # Конец JPEG
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        self.frame = frame
        except Exception as e:
            print(f"Ошибка стрима: {e}")

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()