import base64
import time
from threading import Lock, Thread
import cv2
import numpy
import openai
from PIL import ImageGrab
from cv2 import imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

load_dotenv()


class DesktopScreenshot:
    def __init__(self):
        self.screenshot = None
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.running:
            screenshot = ImageGrab.grab()
            screenshot = cv2.cvtColor(numpy.array(screenshot), cv2.COLOR_RGB2BGR)
            with self.lock:
                self.screenshot = screenshot
            time.sleep(0.1)

    def read(self, encode=False):
        with self.lock:
            screenshot = self.screenshot.copy() if self.screenshot is not None else None
        if encode and screenshot is not None:
            _, buffer = imencode(".jpeg", screenshot)
            return base64.b64encode(buffer)
        return screenshot

    def stop(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()


class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        if not prompt or not image:
           
            return

        image_str = image.decode() if isinstance(image, bytes) else None
        if not image_str:
            print("⚠️ Failed to decode image.")
            return

        try:
            print("Prompt:", prompt)
            response = self.chain.invoke(
                {"prompt": prompt.strip(), "image_base64": image_str},
                config={"configurable": {"session_id": "unused"}},
            ).strip()
            print("Response:", response)
            if response:
                self._tts(response)
        except Exception as e:
            print(f"⚠️ Assistant error: {e}")

    def _tts(self, response):
        try:
            # Generate full audio response
            response_audio = openai.audio.speech.create(
                model="tts-1",
                voice="onyx",
                response_format="pcm",
                input=response,
            )

            audio_data = response_audio.read()

            # Play the full audio once received
            player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)
            player.write(audio_data)
            player.stop_stream()
            player.close()

        except Exception as e:
            print(f"⚠️ TTS error: {e}")


    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions. Your job is to answer 
        questions.



        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. Do not ask questions unless you require additional info.

        Be friendly and helpful. Show some personality.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )
        chain = prompt_template | model | StrOutputParser()
        history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )


class AssistantController:
    def __init__(self):
        self.recognizer = Recognizer()
        self.microphone = Microphone()
        self.model = ChatOpenAI(model="gpt-4o")
        self.assistant = Assistant(self.model)
        self.screen = DesktopScreenshot().start()

    def audio_callback(self, recognizer, audio):
        try:
            prompt = recognizer.recognize_whisper(audio, model="base", language="english")
           
            image = self.screen.read(encode=True)
            self.assistant.answer(prompt, image)
        except UnknownValueError:
            print("❌ Whisper could not understand the audio.")

    def run(self):
        print("🎙️ Assistant is running... Press 'q' or 'Esc' to quit.")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        stop_listening = self.recognizer.listen_in_background(self.microphone, self.audio_callback)

        try:
            while True:
                frame = self.screen.read()
                if frame is not None:
                    cv2.imshow("Screen Capture", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in [27, ord("q")]:
                    print("🛑 Quitting assistant.")
                    break
        finally:
            self.screen.stop()
            cv2.destroyAllWindows()
            stop_listening(wait_for_stop=False)


if __name__ == "__main__":
    controller = AssistantController()
    controller.run()
