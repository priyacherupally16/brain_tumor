import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Step 1: Define Data Paths
folders = {
    'glioma': r'C:\Users\Lenovo\Desktop\miniproject\dataset\Training\glioma',
    'meningioma': r'C:\Users\Lenovo\Desktop\miniproject\dataset\Training\meningioma',
    'pituitary': r'C:\Users\Lenovo\Desktop\miniproject\dataset\Training\pituitary',
    'notumor': r'C:\Users\Lenovo\Desktop\miniproject\dataset\Training\notumor',
}

# Step 2: Load and Preprocess Images
def load_images(folder, label):
    data = []
    for file in os.listdir(folder):
        try:
            img_path = os.path.join(folder, file)
            image = load_img(img_path, target_size=(128, 128))
            image = img_to_array(image)
            image = preprocess_input(image)
            data.append((image, label))
        except:
            continue
    return data

# Step 3: Load data for each class with a label
print("Loading images...")
data = []
data += load_images(folders['glioma'], 0)
data += load_images(folders['meningioma'], 1)
data += load_images(folders['notumor'], 2)
data += load_images(folders['pituitary'], 3)

np.random.shuffle(data)

X = np.array([img for img, label in data])
y = np.array([label for img, label in data])

# Step 4: Feature Extraction with VGG16
print("Extracting features...")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)
features = model.predict(X, batch_size=32)
X_flat = features.reshape(features.shape[0], -1)

# Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# Step 6: Train Random Forest Classifier
print("Training classifier...")
clf = RandomForestClassifier(n_estimators=300, random_state=42)
clf.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (4 Classes)")
plt.tight_layout()
plt.show()


# Step 8: Save Model
with open("rf_model.pkl", "wb") as f:
    pickle.dump(clf, f)

