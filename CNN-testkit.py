# Surface Defect Detection - Fixed for NEU-DET Structure
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

# 1. Upload and extract dataset
from google.colab import files
print("Please upload your 'archive (3).zip' file")
uploaded = files.upload()

# Get the uploaded filename
zip_filename = next(iter(uploaded)) if uploaded else None
if not zip_filename:
    raise FileNotFoundError("No file was uploaded. Please upload your dataset zip file.")

print(f"\nStep 1: Extracting {zip_filename}...")
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall('neu-surface-defect-database')

# 2. Load data from the correct paths
print("\nStep 2: Loading images...")
def load_images():
    base_path = 'neu-surface-defect-database/NEU-DET'
    train_image_path = os.path.join(base_path, 'train/images')
    val_image_path = os.path.join(base_path, 'validation/images')
    
    # Get class names from train directory
    classes = sorted(os.listdir(train_image_path))
    print(f"Found classes: {classes}")
    
    images = []
    labels = []
    
    # Load training images
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(train_image_path, class_name)
        print(f"Loading training images from {class_name}...")
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img.astype('float32') / 255.0
                images.append(img)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
                continue
    
    # Load validation images
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(val_image_path, class_name)
        print(f"Loading validation images from {class_name}...")
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img = img.astype('float32') / 255.0
                images.append(img)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")
                continue
    
    if not images:
        raise ValueError("No images were loaded. Check if the images are in JPG/PNG format.")
    
    return np.array(images), np.array(labels), classes

try:
    X, y, classes = load_images()
    print(f"\nSuccessfully loaded {len(X)} images")
except Exception as e:
    print(f"\nError loading data: {str(e)}")
    print("\nCurrent directory structure:")
    !find neu-surface-defect-database -type d | sort
    raise

# 3. Prepare data
print("\nStep 3: Preparing data...")
y_one_hot = to_categorical(y, num_classes=len(classes))

# Split data (since dataset is already split into train/val, we'll combine and resplit for this example)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_one_hot, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# 4. Data augmentation
print("\nStep 4: Setting up data augmentation...")
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

# 5. Build model
print("\nStep 5: Building model...")
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

# 6. Train model
print("\nStep 6: Training model...")
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 32,
    epochs=30,
    validation_data=val_generator,
    validation_steps=len(X_val) // 32,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# 7. Evaluate
print("\nStep 7: Evaluating model...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# 8. Visualize results
print("\nStep 8: Visualizing predictions...")
def visualize_predictions(num_samples=5):
    plt.figure(figsize=(15, 5))
    y_pred = model.predict(X_test[:num_samples])
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(X_test[i])
        plt.title(f"True: {classes[np.argmax(y_test[i])]}\nPred: {classes[y_pred_classes[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_predictions()

# Classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=classes))

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
