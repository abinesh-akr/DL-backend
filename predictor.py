import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load models
generator = tf.keras.models.load_model('models/generator.keras')
autoencoder = tf.keras.models.load_model('models/autoencoder.keras')

# Load MNIST for reconstruction
(_, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
test_images = (test_images - 127.5) / 127.5  # Normalize to [-1, 1]

@app.route('/generate', methods=['POST'])
def generate_image():
    noise = np.random.normal(0, 1, (1, 100))
    generated_image = generator.predict(noise, verbose=0)[0]
    generated_image = (generated_image * 127.5 + 127.5).astype(np.uint8)
    image_array = generated_image[:, :, 0].tolist()
    return jsonify({'image': image_array})

@app.route('/reconstruct', methods=['POST'])
def reconstruct_image():
    idx = np.random.randint(0, test_images.shape[0])
    test_image = test_images[idx:idx+1]
    reconstructed = autoencoder.predict(test_image, verbose=0)[0]
    reconstructed = (reconstructed * 127.5 + 127.5).astype(np.uint8)
    image_array = reconstructed[:, :, 0].tolist()
    return jsonify({'image': image_array})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))  # Use Render's PORT env var or default to 8000
    app.run(host='0.0.0.0', port=port)
