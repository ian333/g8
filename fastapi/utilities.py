import whisper




async def transcribe(audio):
    model = whisper.load_model('tiny')
    result = model.transcribe(audio)

    return result["text"]



