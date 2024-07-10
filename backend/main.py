
from fastapi import FastAPI, File, UploadFile
import whisper
import os

app = FastAPI()


def recognition(audio_path: str) -> str:
    """
    Данная функция реализует распознавание речи при помощи модели-трансформера Whisper.
    Функция получает путь до временного файла, распознаёт и возвращает речь из него.

    Args:
        audio_path (Str): Путь до временного аудиофайла.
    Returns:
        result['text'] (Str): Распознанный текст из аудиофайла.
    """
    model = whisper.load_model('small')   #загрузка предобученной модели Whisper от OpenAI (есть на выбор tiny, base, small, medium, large)
    audio = audio_path                    #путь до распознаваемого файла
    result = model.transcribe(audio)      #получение результата
    return result['text']                 #возвращение результата


@app.post('/recognize')
async def recognize_audio(file: UploadFile = File(...)) -> dict:
    """
    Данная функция получает на вход аудиофайл, который пробует распознать, и возвращает текстовое представление аудиофайла.

    Args:
        file (UploadFile): Входной файл является загружаемым.
    Return:
        text (Dict): Возвращаем текстовую информацию в виде словаря.
    """
    file_location = f'temp_{file.filename}'          #создаём временное имя для файла
    with open(file_location, 'wb') as file_object:   #открываем файл для записи в бинарном режиме
        file_object.write(file.file.read())          #записываем содержимое полученного файла
    text_from_audio = recognition(file_location)     #получаем текст после распознавания
    if os.path.exists(file_location):                #проверка существования временного файла
        os.remove(file_location)                     #удаление временного файла
    return {'text': text_from_audio}                 #возвращаем результат
