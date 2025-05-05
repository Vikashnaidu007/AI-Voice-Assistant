# 🧠 Voice-Activated Desktop Assistant with Screenshot Context

This Python-based desktop assistant uses voice input, OpenAI GPT-4o, and real-time desktop screenshots to generate witty and context-aware responses — all spoken back to you using text-to-speech (TTS).

## 🎯 Features

- 🖥️ Takes live screenshots of your desktop
- 🎤 Uses your voice to ask questions
- 🧠 Sends the prompt + screenshot to GPT-4o
- 🗣️ Responds via OpenAI's TTS (voice: `onyx`)
- 🪟 Displays real-time screen capture in a window

## 📦 Requirements

- Python 3.8+
- A working microphone
- Compatible with Windows/macOS (tested with `PIL.ImageGrab`)
- OpenAI API Key (via `.env` file)

## 🧰 Dependencies

Install all requirements using:

```bash
pip install -r requirements.txt
