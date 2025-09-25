import tensorflow as tf
import numpy as np
import sounddevice as sd
import sys
import time

# Функция преобразования waveform в спектрограмму по заданному алгоритму
def get_spectrogram(waveform):
    # Вычисляем STFT
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    # Берём абсолютное значение (модуль комплексных чисел)
    spectrogram = tf.abs(spectrogram)
    # Добавляем размерность каналов для совместимости с conv-слоями
    spectrogram = spectrogram[..., tf.newaxis]
    # Изменяем размер до 128x64
    spectrogram = tf.image.resize(spectrogram, [128, 64])
    # Обработка нулевых элементов: заменяем их на "медиану" ненулевых значений
    nonzero_elements = tf.boolean_mask(spectrogram, spectrogram != 0)
    nonzero_median = tf.sort(nonzero_elements)[tf.shape(nonzero_elements)[0] // 4]
    spectrogram = tf.where(spectrogram == 0, nonzero_median, spectrogram)
    return spectrogram

def record_audio(duration=2.6, fs=16000):
    """
    Записывает аудио длительностью duration секунд с частотой дискретизации fs.
    Возвращает 1D numpy-массив аудио данных.
    """
    print(f"Запись аудио длительностью {duration} секунд...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Ждём окончания записи
    return np.squeeze(recording)

def main():
    # Запрос пути к модели у пользователя
    # mse_loss = tf.keras.losses.MeanSquaredError()
    loss_ssim = tf.image.ssim
    model_path = r"new_model4.h5".strip()
    # model_path = input("Введите путь к модели TensorFlow Keras: ").strip()
    try:
        # Загрузка модели
        model = tf.keras.models.load_model(model_path)

        classification_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer('classification_output').output
        )
        # Компилируем подмодель с нужными функцией потерь и метриками
        classification_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )
        encoder_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer('reconstruction_output').output
        )
        encoder_model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
        )
        print("Модель успешно загружена.")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        sys.exit(1)






    fs = 16000     # Частота дискретизации
    duration = 2.6 # Длительность записи (секунд)
    label_names = ["Catch", "Gun", "Index", "Like", "Relax", "Rock"]

    print("Начало распознавания голосовых команд. Для выхода нажмите Ctrl+C.")
    try:
        while True:
            waveform = record_audio(duration=duration, fs=fs)
            # Приводим аудио к формату tf.float32 и создаём тензор
            waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
            # Получаем спектрограмму
            spectrogram = get_spectrogram(waveform_tensor)
            # Добавляем размерность батча
            spectrogram = tf.expand_dims(spectrogram, axis=0)  # Форма: [1, 128, 64, 1]


            prediction = classification_model(spectrogram)
            for j in range(len(label_names)):
                print(label_names[j], ":", float(tf.nn.softmax(prediction[0])[j]), end=" ")
            print()
            print(label_names[tf.argmax(prediction[0])], float(tf.nn.softmax(prediction[0])[tf.argmax(prediction[0])]))
            print(float(loss_ssim(spectrogram, encoder_model(spectrogram), max_val=1.0)))
            # Небольшая задержка (можно изменить или удалить)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nПриложение завершает работу.")

if __name__ == '__main__':
    main()
