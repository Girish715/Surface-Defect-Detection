# Complete Surface Defect Detection with Enhanced Visualizations
import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from google.colab import files

# 1. Data Loading with Progress Tracking
def load_data_with_progress():
    print("📁 Upload your dataset zip file")
    uploaded = files.upload()
    if not uploaded:
        raise FileNotFoundError("No file uploaded")
    
    zip_filename = next(iter(uploaded))
    print("⚙️ Extracting files...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall('defect-data')
    
    base_path = 'defect-data/NEU-DET'
    classes = sorted(os.listdir(os.path.join(base_path, 'train/images')))
    
    X = []
    y = []
    img_size = (128, 128)
    
    plt.figure(figsize=(12, 6))
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(base_path, 'train/images', class_name)
        images = os.listdir(class_path)[:300]  # Limit samples
        
        # Plot sample images
        plt.subplot(2, 3, class_idx+1)
        sample_img = cv2.imread(os.path.join(class_path, images[0]))
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        plt.imshow(cv2.resize(sample_img, (100, 100)))
        plt.title(f"{class_name}\n{len(images)} images")
        plt.axis('off')
        
        # Load all images
        for img_file in images:
            img = cv2.imread(os.path.join(class_path, img_file))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(class_idx)
    
    plt.suptitle("Dataset Samples by Class", y=1.05, fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return np.array(X), np.array(y), classes

# 2. Load and prepare data
X, y, classes = load_data_with_progress()
X = X.astype('float32') / 255.0
y = to_categorical(y, num_classes=len(classes))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 3. Enhanced Data Augmentation Visualization
print("\n🎨 Data Augmentation Examples:")
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

plt.figure(figsize=(15, 5))
for i in range(5):  # Show 5 augmented examples
    augmented = train_datagen.flow(np.array([X_train[0]]), batch_size=1)
    plt.subplot(1, 5, i+1)
    plt.imshow(augmented.__next__()[0])  # Changed from .next() to .__next__()
    plt.axis('off')
    if i == 2:
        plt.title("Augmented Samples", pad=20)
plt.show()

# 4. Model Architecture
def build_visual_model():
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(128, 128, 3),
        alpha=0.35
    )
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(len(classes), activation='softmax')
    ])
    
    # Visualize model architecture
    print("\n🧠 Model Architecture:")
    model.summary()
    
    return model

model = build_visual_model()
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy', 
                      tf.keras.metrics.Precision(name='precision'),
                      tf.keras.metrics.Recall(name='recall')])

# 5. Training with LR Scheduling
def lr_schedule(epoch, lr):
    if epoch > 5:
        return lr * 0.9  # Reduce LR after 5 epochs
    return lr

history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train)//32,
    epochs=15,
    validation_data=(X_val, y_val),
    callbacks=[
        EarlyStopping(patience=3),
        LearningRateScheduler(lr_schedule)
    ],
    verbose=1
)

# 6. Enhanced Training Visualization
plt.figure(figsize=(18, 6))

# Accuracy Plot
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy', pad=10)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Loss Plot
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss', pad=10)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

# Learning Rate Plot
plt.subplot(1, 3, 3)
lr_history = [lr_schedule(i, 0.001) for i in range(15)]
plt.plot(lr_history, marker='o')
plt.title('Learning Rate Schedule', pad=10)
plt.ylabel('Learning Rate')
plt.xlabel('Epoch')

plt.tight_layout()
plt.show()

# 7. Comprehensive Evaluation
print("\n📊 Model Evaluation:")
test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)

# Metrics Bar Chart
metrics = ['Accuracy', 'Precision', 'Recall']
values = [test_acc, test_precision, test_recall]

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FF9800'])
plt.ylim(0, 1.1)
plt.title('Test Set Performance Metrics', pad=15)
plt.ylabel('Score')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom')

plt.show()

# Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix', pad=15)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 8. Prediction with Class Distribution
def analyze_with_distribution():
    print("\n🔍 Upload an image for analysis")
    uploaded = files.upload()
    if not uploaded:
        return
    
    file = next(iter(uploaded))
    img = cv2.imdecode(np.frombuffer(uploaded[file], np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    display_img = cv2.resize(img, (200, 200))
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img.astype('float32')/255.0, axis=0)
    
    pred = model.predict(img, verbose=0)
    class_idx = np.argmax(pred)
    
    plt.figure(figsize=(15, 5))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(display_img)
    plt.title("Uploaded Image", pad=10)
    plt.axis('off')
    
    # Class Probabilities
    plt.subplot(1, 3, 2)
    bars = plt.barh(classes, pred[0]*100, color='skyblue')
    plt.xlabel('Confidence (%)')
    plt.title('Defect Probabilities', pad=10)
    
    # Highlight max probability
    bars[class_idx].set_color('#FF5722')
    
    # Training Distribution
    plt.subplot(1, 3, 3)
    class_dist = np.sum(y_train, axis=0)
    plt.barh(classes, class_dist, color='lightgreen')
    plt.xlabel('Count')
    plt.title('Training Class Distribution', pad=10)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🔎 Result: {classes[class_idx]} ({np.max(pred)*100:.1f}% confidence)")

analyze_with_distribution()
