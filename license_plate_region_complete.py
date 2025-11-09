"""
License Plate Region Detection - Complete Pipeline
Combines data loading, preprocessing, model training, and inference
"""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ==================== DATA LOADING & PREPROCESSING ====================

class LicensePlateDataLoader:
    """Handles loading and preprocessing of license plate dataset"""
    
    def __init__(self, images_dir, annotations_dir, img_size=(224, 224)):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.img_size = img_size
        
    def parse_annotation(self, xml_file):
        """Parse XML annotation file and extract bounding box coordinates"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # Get bounding box coordinates
        boxes = []
        for obj in root.findall('object'):
            if obj.find('name').text == 'license_plate':
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
        
        return boxes, width, height
    
    def normalize_bbox(self, bbox, orig_width, orig_height):
        """Normalize bounding box coordinates to [0, 1]"""
        xmin, ymin, xmax, ymax = bbox
        return [
            xmin / orig_width,
            ymin / orig_height,
            xmax / orig_width,
            ymax / orig_height
        ]
    
    def load_data(self):
        """Load all images and annotations"""
        images = []
        bboxes = []
        
        print("Loading dataset...")
        for img_file in os.listdir(self.images_dir):
            if not img_file.endswith(('.jpg', '.png', '.jpeg')):
                continue
                
            # Load image
            img_path = os.path.join(self.images_dir, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get annotation
            xml_file = os.path.join(
                self.annotations_dir, 
                img_file.replace('.jpg', '.xml').replace('.png', '.xml')
            )
            
            if not os.path.exists(xml_file):
                continue
            
            try:
                boxes, orig_w, orig_h = self.parse_annotation(xml_file)
            except:
                continue
            
            if len(boxes) == 0:
                continue
            
            # Resize image
            img_resized = cv2.resize(img, self.img_size)
            
            # Normalize first bounding box
            norm_bbox = self.normalize_bbox(boxes[0], orig_w, orig_h)
            
            images.append(img_resized)
            bboxes.append(norm_bbox)
        
        print(f"Loaded {len(images)} images successfully")
        return np.array(images), np.array(bboxes)
    
    def visualize_sample(self, img, bbox, title='License Plate Detection'):
        """Visualize image with bounding box"""
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        
        # Denormalize bbox for visualization
        h, w = img.shape[:2]
        xmin, ymin, xmax, ymax = bbox
        xmin, xmax = int(xmin * w), int(xmax * w)
        ymin, ymax = int(ymin * h), int(ymax * h)
        
        # Draw rectangle
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                            fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
        plt.axis('off')
        plt.title(title)
        plt.show()


# ==================== MODEL ARCHITECTURE & TRAINING ====================

class LicensePlateDetector:
    """CNN-based license plate bounding box detector"""
    
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build a simple CNN for bounding box regression"""
        model = keras.Sequential([
            # Convolutional layers
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer: 4 values for bbox [xmin, ymin, xmax, ymax]
            layers.Dense(4, activation='sigmoid')  # sigmoid to keep values in [0,1]
        ])
        
        self.model = model
        print("Model architecture built successfully")
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile model with appropriate loss and optimizer"""
        if self.model is None:
            self.build_model()
            
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',  # Mean Squared Error for regression
            metrics=['mae']  # Mean Absolute Error
        )
        print("Model compiled successfully")
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train
        print(f"\nStarting training for {epochs} epochs...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        print("\nTraining completed!")
        return self.history
    
    def plot_training_history(self):
        """Plot training and validation loss"""
        if self.history is None:
            print("No training history available")
            return
            
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='Train MAE')
        plt.plot(self.history.history['val_mae'], label='Val MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath='license_plate_detector.h5'):
        """Save the trained model"""
        if self.model is None:
            print("No model to save")
            return
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='license_plate_detector.h5'):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


# ==================== INFERENCE & PREDICTION ====================

class LicensePlatePredictor:
    """Handles inference and prediction on new images"""
    
    def __init__(self, model_path=None, model=None, img_size=(224, 224)):
        self.img_size = img_size
        
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = keras.models.load_model(model_path)
        else:
            raise ValueError("Either model_path or model must be provided")
    
    def preprocess_image(self, image_path):
        """Load and preprocess image for prediction"""
        # Read image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Store original dimensions
        orig_h, orig_w = img_rgb.shape[:2]
        
        # Resize for model
        img_resized = cv2.resize(img_rgb, self.img_size)
        img_normalized = img_resized / 255.0
        
        # Add batch dimension
        img_input = np.expand_dims(img_normalized, axis=0)
        
        return img_input, img_rgb, (orig_w, orig_h)
    
    def predict(self, image_path):
        """Predict bounding box for license plate"""
        img_input, img_orig, (orig_w, orig_h) = self.preprocess_image(image_path)
        
        # Predict normalized bbox
        bbox_norm = self.model.predict(img_input, verbose=0)[0]
        
        # Denormalize to original image dimensions
        xmin, ymin, xmax, ymax = bbox_norm
        bbox_pixel = [
            int(xmin * orig_w),
            int(ymin * orig_h),
            int(xmax * orig_w),
            int(ymax * orig_h)
        ]
        
        return bbox_pixel, img_orig
    
    def visualize_prediction(self, image_path, save_path=None):
        """Predict and visualize result"""
        bbox, img = self.predict(image_path)
        xmin, ymin, xmax, ymax = bbox
        
        # Draw bounding box
        img_with_bbox = img.copy()
        cv2.rectangle(img_with_bbox, (xmin, ymin), (xmax, ymax), 
                     (255, 0, 0), 3)
        
        # Add label
        cv2.putText(img_with_bbox, 'License Plate', 
                   (xmin, ymin - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.9, (255, 0, 0), 2)
        
        # Display
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_with_bbox)
        plt.title('Detection Result')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Result saved to {save_path}")
        
        plt.show()
        
        return bbox
    
    def extract_license_plate(self, image_path):
        """Extract the license plate region from image"""
        bbox, img = self.predict(image_path)
        xmin, ymin, xmax, ymax = bbox
        
        # Ensure valid crop coordinates
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax = min(img.shape[1], xmax)
        ymax = min(img.shape[0], ymax)
        
        # Crop license plate region
        license_plate = img[ymin:ymax, xmin:xmax]
        
        return license_plate
    
    def evaluate_iou(self, pred_bbox, true_bbox):
        """Calculate Intersection over Union (IoU) metric"""
        # Calculate intersection
        x_left = max(pred_bbox[0], true_bbox[0])
        y_top = max(pred_bbox[1], true_bbox[1])
        x_right = min(pred_bbox[2], true_bbox[2])
        y_bottom = min(pred_bbox[3], true_bbox[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
        true_area = (true_bbox[2] - true_bbox[0]) * (true_bbox[3] - true_bbox[1])
        union = pred_area + true_area - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou


# ==================== MAIN PIPELINE ====================

def run_complete_pipeline(images_dir, annotations_dir, test_image_path=None):
    """
    Complete pipeline: Load data -> Train model -> Test prediction
    
    Args:
        images_dir: Path to training images
        annotations_dir: Path to XML annotations
        test_image_path: Path to test image (optional)
    """
    
    print("="*60)
    print("LICENSE PLATE REGION DETECTION - COMPLETE PIPELINE")
    print("="*60)
    
    # Step 1: Load and preprocess data
    print("\n[STEP 1] Loading and preprocessing data...")
    loader = LicensePlateDataLoader(images_dir, annotations_dir)
    images, bboxes = loader.load_data()
    
    if len(images) == 0:
        print("Error: No images loaded. Check your data paths.")
        return
    
    print(f"Dataset loaded: {len(images)} images")
    print(f"Image shape: {images[0].shape}")
    print(f"Bbox shape: {bboxes[0].shape}")
    
    # Normalize images
    images = images / 255.0
    
    # Visualize a sample
    print("\nVisualizing sample image...")
    loader.visualize_sample((images[0] * 255).astype(np.uint8), bboxes[0])
    
    # Step 2: Split data
    print("\n[STEP 2] Splitting data into train/validation/test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, bboxes, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Step 3: Build and train model
    print("\n[STEP 3] Building and training model...")
    detector = LicensePlateDetector(input_shape=(224, 224, 3))
    detector.build_model()
    detector.model.summary()
    
    detector.compile_model(learning_rate=0.001)
    
    history = detector.train(
        X_train, y_train, 
        X_val, y_val,
        epochs=50,
        batch_size=16
    )
    
    # Plot training history
    detector.plot_training_history()
    
    # Save model
    detector.save_model('license_plate_detector.h5')
    
    # Step 4: Evaluate on test set
    print("\n[STEP 4] Evaluating on test set...")
    predictor = LicensePlatePredictor(model=detector.model)
    
    ious = []
    for i in range(min(len(X_test), 10)):  # Test on first 10 samples
        # Get prediction (need to denormalize)
        img_resized = (X_test[i] * 255).astype(np.uint8)
        
        # For simplicity, assume test images are already at 224x224
        # In real scenario, you'd need original image dimensions
        pred_bbox_norm = detector.model.predict(np.expand_dims(X_test[i], axis=0), verbose=0)[0]
        true_bbox_norm = y_test[i]
        
        # Calculate IoU on normalized coordinates
        iou = predictor.evaluate_iou(pred_bbox_norm, true_bbox_norm)
        ious.append(iou)
    
    avg_iou = np.mean(ious)
    print(f"\nAverage IoU on test samples: {avg_iou:.3f}")
    
    # Step 5: Test on custom image (if provided)
    if test_image_path and os.path.exists(test_image_path):
        print(f"\n[STEP 5] Testing on custom image: {test_image_path}")
        bbox = predictor.visualize_prediction(test_image_path, save_path='result.png')
        print(f"Predicted bounding box: {bbox}")
        
        # Extract license plate
        plate = predictor.extract_license_plate(test_image_path)
        plt.figure(figsize=(6, 3))
        plt.imshow(plate)
        plt.title('Extracted License Plate')
        plt.axis('off')
        plt.show()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return detector, predictor


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    # Configuration
    IMAGES_DIR = "data/images"
    ANNOTATIONS_DIR = "data/annotations"
    TEST_IMAGE = "test_images/car1.jpg"
    
    # Run complete pipeline
    detector, predictor = run_complete_pipeline(
        IMAGES_DIR, 
        ANNOTATIONS_DIR,
        TEST_IMAGE
    )
    
    # Additional testing (optional)
    print("\n\nAdditional Testing:")
    print("-" * 40)
    
    # Test on multiple images
    test_images = [
        "test_images/car1.jpg",
        "test_images/car2.jpg",
        "test_images/car3.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\nTesting: {img_path}")
            bbox = predictor.visualize_prediction(img_path)
            print(f"Bbox: {bbox}")
