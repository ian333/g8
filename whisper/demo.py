#aqui ira el codigo de whisper

from IPython.display import HTML, Audio
from google.colab.output import eval_js
from base64 import b64decode
import numpy as np

#estas son las librerias para trabajar con archivos .wav
from scipy.io.wavfile import read as wav_read
import io
import ffmpeg


Tarea = "Transcript to Language" #@param ["Transcript to Language", "Translate to English"]

#estas son las librerias especiales
import numpy as np
import whisper
from scipy.io.wavfile import write
from IPython.display import clear_output


def get_audio():
  display(HTML(AUDIO_HTML))
  data = eval_js("data")
  binary = b64decode(data.split(',')[1])
  
  process = (ffmpeg
    .input('pipe:0')
    .output('pipe:1', format='mp3')
    .run_async(pipe_stdin=False, pipe_stdout=True, pipe_stderr=True, quiet=True, overwrite_output=True)
  )
  output, err = process.communicate(input=binary)
  