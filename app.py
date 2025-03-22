from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Create an uploads folder if it doesn't exist
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Load the saved model (native Keras format)
model = load_model("rice_classifier_model.keras")
print("Model loaded successfully!")

# Define class labels (ensure these match your training labels)
class_labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the input image
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    filename = None
    confidence = None
    error = None
    
    if request.method == "POST":
        # Check if the post request has the file part
        if 'image' not in request.files:
            error = "No file part"
        else:
            image_file = request.files["image"]
            # Check if the user selected a file
            if image_file.filename == '':
                error = "No selected file"
            elif image_file and allowed_file(image_file.filename):
                # Secure the filename to prevent security issues
                filename = secure_filename(image_file.filename)
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(img_path)
                
                try:
                    # Preprocess the image and make a prediction
                    processed_image = preprocess_image(img_path)
                    pred = model.predict(processed_image)
                    predicted_class = np.argmax(pred)
                    prediction = class_labels[predicted_class]
                    confidence = float(pred[0][predicted_class]) * 100
                except Exception as e:
                    error = f"Error processing image: {str(e)}"
                    filename = None
            else:
                error = "File type not allowed. Please upload a JPG, JPEG or PNG file."
    
    # Render the home page with the form and, if available, the prediction
    return render_template(
        "index.html", 
        prediction=prediction, 
        filename=filename,
        confidence=confidence,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)