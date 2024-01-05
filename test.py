import cv2
from keras.models import load_model
import numpy as np

# Muat model ekspresi
model = load_model('emotion_model.hdf5')

# Daftar label ekspresi
expressions = ['Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Netral']

# Mulai streaming video dari kamera
cap = cv2.VideoCapture(0)

while True:
    # Baca sebuah frame dari kamera
    ret, frame = cap.read()

    # Konversi frame ke skala keabuan
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah menggunakan Haarcascades atau metode lainnya
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Potong wajah dari frame
        face_roi = gray[y:y+h, x:x+w]
        
        # Ubah ukuran wajah sesuai bentuk masukan yang diharapkan oleh model (misalnya, 64x64)
        face_roi = cv2.resize(face_roi, (64, 64))
        face_roi = face_roi.astype('float') / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)

        # Prediksi ekspresi
        emotion = expressions[np.argmax(model.predict(face_roi))]

        # Tampilkan ekspresi yang diprediksi pada frame
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow('Deteksi Ekspresi', frame)

    # Berhenti jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bebaskan kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()
