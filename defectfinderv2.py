import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from google.colab import files

# 1. Fast Data Loading
def load_data_fast():
    # Upload and extract
    uploaded = files.upload()
    if not uploaded:
        raise FileNotFoundError("No file uploaded")
    
    zip_filename = next(iter(uploaded))
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall('defect-data')
    
    # Load images quickly with OpenCV
    base_path = 'defect-data/NEU-DET'
    classes = sorted(os.listdir(os.path.join(base_path, 'train/images')))
    
    X = []
    y = []
    img_size = (128, 128)  # Smaller size for faster processing
    
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(base_path, 'train/images', class_name)
        for img_file in os.listdir(class_path)[:300]:  # Limit samples per class
            img = cv2.imread(os.path.join(class_path, img_file))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(class_idx)
    
    return np.array(X), np.array(y), classes

# 2. Load and prepare data
X, y, classes = load_data_fast()
X = X.astype('float32') / 255.0
y = tf.keras.utils.to_categorical(y, num_classes=len(classes))

# Use smaller test size for faster evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 3. Fast Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# 4. Optimized MobileNetV2 Model (best speed/accuracy tradeoff)
def build_fast_model():
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(128, 128, 3),
        alpha=0.35  # Smaller version
    )
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(len(classes), activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 5. Fast Training
model = build_fast_model()

history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train)//32,
    epochs=15,  # Fewer epochs
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(patience=3)],
    verbose=1
)

# 6. Quick Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# 7. Fast Prediction
def quick_predict():
    print("Upload an image for quick defect analysis")
    uploaded = files.upload()
    if not uploaded:
        return
    
    file = next(iter(uploaded))
    img = cv2.imdecode(np.frombuffer(uploaded[file], np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img.astype('float32')/255.0, axis=0)
    
    pred = model.predict(img, verbose=0)
    class_idx = np.argmax(pred)
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img[0])
    plt.title("Uploaded Image")
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.barh(classes, pred[0]*100)
    plt.xlabel('Confidence (%)')
    plt.title('Defect Probabilities')
    plt.tight_layout()
    plt.show()
    
    print(f"\nPredicted Defect: {classes[class_idx]} ({np.max(pred)*100:.1f}% confidence)")

quick_predict()
