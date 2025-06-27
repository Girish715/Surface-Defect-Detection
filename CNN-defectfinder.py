# Enhanced Surface Defect Detection with Image Upload Functionality
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from google.colab import files
from IPython.display import display, HTML

# 1. Upload and extract dataset
print("Please upload your 'archive(3).zip' file")
uploaded = files.upload()
zip_filename = next(iter(uploaded)) if uploaded else None
if not zip_filename:
    raise FileNotFoundError("No file was uploaded. Please upload your dataset zip file.")

print(f"\nStep 1: Extracting {zip_filename}...")
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall('neu-surface-defect-database')

# 2. Load data from the correct paths
def load_images():
    base_path = 'neu-surface-defect-database/NEU-DET'
    train_image_path = os.path.join(base_path, 'train/images')
    val_image_path = os.path.join(base_path, 'validation/images')
    
    classes = sorted(os.listdir(train_image_path))
    defect_descriptions = {
        'crazing': 'Network of fine cracks on the surface',
        'inclusion': 'Foreign materials embedded in the surface',
        'patches': 'Localized surface imperfections',
        'pitted_surface': 'Small holes or pits on the surface',
        'rolled-in_scale': 'Oxide scale rolled into the surface',
        'scratches': 'Linear surface marks caused by abrasion'
    }
    
    images = []
    labels = []
    
    # Load training images
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(train_image_path, class_name)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img.astype('float32') / 255.0
                images.append(img)
                labels.append(class_idx)
    
    # Load validation images
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(val_image_path, class_name)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img.astype('float32') / 255.0
                images.append(img)
                labels.append(class_idx)
    
    return np.array(images), np.array(labels), classes, defect_descriptions

X, y, classes, defect_descriptions = load_images()
print(f"\nSuccessfully loaded {len(X)} images")

# 3. Prepare data
y_one_hot = to_categorical(y, num_classes=len(classes))
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# 4. Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 5. Build and train model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    epochs=30,
    validation_data=(X_val, y_val),
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# 6. Enhanced prediction function for uploaded images
def analyze_surface_defect():
    print("\nUpload an image for surface defect analysis")
    uploaded = files.upload()
    for filename in uploaded.keys():
        # Process uploaded image
        img = cv2.imdecode(np.frombuffer(uploaded[filename], np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        display_img = cv2.resize(img, (300, 300))
        
        # Prepare image for prediction
        img_for_pred = cv2.resize(img, (224, 224))
        img_for_pred = img_for_pred.astype('float32') / 255.0
        img_for_pred = np.expand_dims(img_for_pred, axis=0)
        
        # Make prediction
        pred = model.predict(img_for_pred)
        pred_class = classes[np.argmax(pred)]
        confidence = np.max(pred) * 100
        description = defect_descriptions.get(pred_class, "No description available")
        
        # Display results
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(display_img)
        plt.title("Uploaded Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.barh(classes, pred[0] * 100, color='skyblue')
        plt.xlabel('Confidence (%)')
        plt.title('Defect Probability')
        plt.tight_layout()
        plt.show()
        
        # Generate detailed report
        report = f"""
        <div style='border:2px solid #4CAF50; padding:20px; border-radius:10px; margin:20px;'>
            <h2 style='color:#4CAF50;'>Surface Defect Analysis Report</h2>
            <p><strong>Detected Defect:</strong> {pred_class}</p>
            <p><strong>Confidence:</strong> {confidence:.2f}%</p>
            <p><strong>Description:</strong> {description}</p>
            <p><strong>Recommended Action:</strong> {get_recommendation(pred_class)}</p>
        </div>
        """
        display(HTML(report))

def get_recommendation(defect_type):
    recommendations = {
        'crazing': 'Monitor for propagation. Consider stress relief treatments.',
        'inclusion': 'Remove contaminated material. Improve filtration during production.',
        'patches': 'Surface grinding or polishing may be required.',
        'pitted_surface': 'Evaluate for corrosion. Consider protective coatings.',
        'rolled-in_scale': 'Descaling treatment recommended. Improve rolling process.',
        'scratches': 'Assess depth. Minor scratches can be polished out.'
    }
    return recommendations.get(defect_type, 'Further inspection recommended.')

# 7. Run the analysis
print("\nModel training complete. You can now analyze surface defects.")
print("Defect types this model can detect:")
for i, defect in enumerate(classes, 1):
    print(f"{i}. {defect}: {defect_descriptions[defect]}")

analyze_surface_defect()
