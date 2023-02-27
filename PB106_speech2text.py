import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import logging

import warnings
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

czech_model = "arampacha/wav2vec2-large-xlsr-czech"
smaller_model = ".\\s2t_smaller"
bigger_model = ".\\s2t_bigger"
audio = ".\\audio.mp3"
audio2 = ".\\audio2.mp3"
audio3 = ".\\audio3.mp3"


def speech_to_text_with_wav2vec(language_model, input_file):
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(language_model)
    model = Wav2Vec2ForCTC.from_pretrained(language_model)
    audio, rate = librosa.load(input_file, sr=16000)
    input_values = tokenizer(audio, return_tensors='pt').input_values
    logits = model(input_values).logits
    prediction = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(prediction)[0]
    print(transcription)


print("Převod řeči na text pomocí českého modelu "
      "arampacha/wav2vec2-large-xlsr-czech:")
speech_to_text_with_wav2vec(czech_model, audio)
speech_to_text_with_wav2vec(czech_model, audio2)
speech_to_text_with_wav2vec(czech_model, audio3)

print("\nPřevod řeči na text pomocí vlastního natrénovaného modelu "
      "s 500 train+validation daty, 250 test daty a 40 epochami:")
speech_to_text_with_wav2vec(smaller_model, audio)
speech_to_text_with_wav2vec(smaller_model, audio2)
speech_to_text_with_wav2vec(smaller_model, audio3)

print("\nPřevod řeči na text pomocí vlastního natrénovaného modelu "
      "s 2000 train+validation daty, 1000 test daty a 50 epochami:")
speech_to_text_with_wav2vec(bigger_model, audio)
speech_to_text_with_wav2vec(bigger_model, audio2)
speech_to_text_with_wav2vec(bigger_model, audio3)

