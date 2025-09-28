import openai
import telepot
import os

from telepot.loop import MessageLoop

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta

from chatterbox.tts import ChatterboxTTS

# -------------------------------------------------------------------------------------------------

# Detect device (Mac with M1/M2/M3/M4)

device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

torch_load_original = torch.load

def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load

model = ChatterboxTTS.from_pretrained(device=device)

client = openai.Client()

# -------------------------------------------------------------------------------------------------

def generate_voice(out_file, reference_file):
    request = "Generate a sentence to remind me about my QA course homework."
    reminder = client.responses.create(model="gpt-5", input=request)

    reminder_text = reminder.output_text
    wav = model.generate(reminder_text, audio_prompt_path=reference_file)

    if os.path.exists(out_file):
        os.remove(out_file)

    ta.save(out_file, wav, model.sr)

def handle(msg):
    REF_AUDIO = "ref_audio.mp3"
    RESULT_AUDIO = "voice.wav"

    chat_id = msg['chat']['id']
    command = msg['text']

    if command == '/voice':
        generate_voice(RESULT_AUDIO, REF_AUDIO)

        with open(RESULT_AUDIO, 'rb') as voice_file:
            bot.sendVoice(chat_id, voice=voice_file)

# -------------------------------------------------------------------------------------------------

openai.api_key = os.environ['OPENAI_API_KEY']
token = os.environ['BOT_TOKEN']
bot = telepot.Bot(token)

MessageLoop(bot, handle).run_forever()