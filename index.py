import os
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.imagenet_utils import preprocess_input 

app = Flask(__name__)

# Muat model H5
MODEL_PATH = './model_buah.h5'
model = tf.keras.models.load_model(MODEL_PATH)

def prepare_image(img_path):
    """Mempersiapkan gambar untuk klasifikasi"""
    img = image.load_img(img_path, target_size=(126, 126))  # Sesuaikan ukuran dengan model Anda
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.imagenet_utils.preprocess_input(img_array)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk klasifikasi gambar"""
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar'}), 400
    
    file = request.files['file']
    
    # Simpan file sementara
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    
    try:
        # Persiapkan gambar
        processed_image = prepare_image(file_path)
        
        # Prediksi
        predictions = model.predict(processed_image)
        
        # Dapatkan kelas dengan probabilitas tertinggi
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        class_names = {0: 'jeruk', 1: 'apel', 2: 'pisang', 3: 'pisang', 4: 'pisang', 5: 'pisang', 6: 'pisang', 7: 'pisang', 8: 'pisang'}  # Sesuaikan dengan kelas yang ada di model Anda
        predicted_class_name = class_names.get(predicted_class, 'Unknown')
        condition_names = {0: 'Fresh', 1: 'Mild', 2: 'Rotten'}  # Sesuaikan dengan kelas yang ada di model Anda
        predicted_condition_name = condition_names.get(confidence, 'Unknown')
        # Hapus file sementara
        os.remove(file_path)
        
        return jsonify({
            'class': predicted_class_name,
            'condition': predicted_condition_name
        })
    
    except Exception as e:
        # Hapus file jika terjadi error
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Buat folder uploads jika belum ada
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)