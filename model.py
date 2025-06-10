import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from scipy.signal import stft
from scipy.stats import kurtosis, skew
import shap
import time
import os

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to load data from a CSV file
def load_data(filename):
    try:
        df = pd.read_csv(filename, header=None, delimiter='\s+')
        print(f"Loading {filename}: {df.shape} columns")
        data = df.iloc[:, 0:4].values
        if data.shape[1] != 4:
            raise ValueError(f"Expected 4 columns (motor, x, y, z), got {data.shape[1]} in {filename}")
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        raise

# Data augmentation functions
def time_shift(data, max_shift=3):
    shift = np.random.randint(-max_shift, max_shift)
    if shift > 0:
        return np.pad(data, ((shift, 0), (0, 0)), mode='constant')[:-shift]
    else:
        return np.pad(data, ((0, -shift), (0, 0)), mode='constant')[-shift:]

def scale_signal(data, scale_range=(0.97, 1.03)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return data * scale

def jitter(data, jitter_factor=0.03):
    jitter = np.random.normal(0, jitter_factor, data.shape)
    return data + jitter

def add_noise(data, noise_factor=0.03):
    noise = np.random.normal(0, noise_factor, data.shape)
    return data + noise

def augment_data(data):
    data = time_shift(data)
    data = scale_signal(data)
    data = jitter(data)
    data = add_noise(data)
    return data

# Function to create windows from the data
def create_windows(data, window_size=128, step_size=64):
    if data.shape[1] != 4:
        raise ValueError(f"Data must have 4 columns, got {data.shape[1]}")
    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[i:i + window_size])
    windows = np.array(windows)
    if windows.size == 0:
        raise ValueError("No windows created; check data length and window_size")
    return windows

# Function to compute STFT for each axis
def compute_stft(data, nperseg=64):
    if data.shape != (128, 4):
        raise ValueError(f"Expected window shape (128, 4), got {data.shape}")
    stft_results = []
    for i in range(data.shape[1]):
        f, t, Zxx = stft(data[:, i], nperseg=nperseg, noverlap=nperseg//2)
        stft_results.append(np.abs(Zxx))
    if not stft_results:
        raise ValueError("No STFT results computed; check input data")
    return np.stack(stft_results, axis=-1)

# Function to normalize STFT data
def normalize_data(data):
    return (data - np.mean(data)) / (np.std(data) + 1e-8)

# Function to extract handcrafted features for SVM and RF
def extract_time_features(data):
    features = []
    for window in data:
        feat = {
            'rms': np.sqrt(np.mean(window**2, axis=0)),
            'kurtosis': kurtosis(window, axis=0),
            'skewness': skew(window, axis=0),
            'mean': np.mean(window, axis=0),
            'std': np.std(window, axis=0)
        }
        features.append(np.concatenate([v for v in feat.values()]))
    return np.array(features)

# Load and process data
files = ['Health_20_0.csv', 'Miss_20_0.csv', 'Health_30_2.csv', 'Miss_30_2.csv']
labels = [0, 1, 0, 1]
X_data, y_data, X_windows_data = [], [], []

for file, label in zip(files, labels):
    raw_data = load_data(file)
    augmented_data = augment_data(raw_data)
    windows = create_windows(augmented_data)
    X_windows_data.append(windows)
    stft_data = np.array([compute_stft(window) for window in windows])
    stft_data = normalize_data(stft_data)
    X_data.append(stft_data)
    y_data.append(np.full(stft_data.shape[0], label))

# Combine data
X_combined = np.concatenate(X_data, axis=0)
y_combined = np.concatenate(y_data, axis=0)
X_windows_combined = np.concatenate(X_windows_data, axis=0)

print(f"X_combined shape: {X_combined.shape}, y_combined shape: {y_combined.shape}")
print(f"X_windows_combined shape: {X_windows_combined.shape}")

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_combined), y=y_combined)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Class weights: {class_weight_dict}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)
X_train_windows, X_test_windows, _, _ = train_test_split(X_windows_combined, y_combined, test_size=0.2, random_state=42)

# CNN Model with Residual Connection
def build_cnn_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(x)
    residual = layers.Conv2D(64, (1, 1), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    return model

# 1D-CNN Model
def build_1d_cnn_model(input_shape=(128, 4)):
    model = keras.Sequential([
        layers.Conv1D(32, 3, activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_callback = keras.callbacks.LearningRateScheduler(lambda epoch, lr: lr * 0.8 if epoch > 10 else lr)

# K-fold cross-validation for CNN
kfold = KFold(n_splits=2, shuffle=True, random_state=42)
results = {
    'Model': [],
    'Accuracy': [],
    'AUC': [],
    'Recall': [],
    'FPR': [],
    'MCC': [],
    'Training Time (s)': []
}

# Train and evaluate CNN
fold_accuracies, fold_aucs = [], []
start_time = time.time()
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
    print(f"\nTraining CNN Fold {fold + 1}/5")
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    model = build_cnn_model(X_tr.shape[1:])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    history = model.fit(X_tr, y_tr, epochs=20, batch_size=32, validation_data=(X_val, y_val),
                        callbacks=[early_stopping, lr_callback], class_weight=class_weight_dict, verbose=0)
    val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
    fold_accuracies.append(val_acc)
    fold_aucs.append(val_auc)
    print(f"Fold {fold + 1}: Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")

cnn_time = time.time() - start_time
print(f"\nCNN Cross-Validation: Accuracy: {np.mean(fold_accuracies):.4f} (±{np.std(fold_accuracies):.4f}), AUC: {np.mean(fold_aucs):.4f} (±{np.std(fold_aucs):.4f})")

# Train final CNN
model = build_cnn_model(X_train.shape[1:])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2,
                    callbacks=[early_stopping, lr_callback], class_weight=class_weight_dict, verbose=0)

# Evaluate CNN on test set
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
y_pred = (model.predict(X_test) > 0.5).astype(int)
report = classification_report(y_test, y_pred, target_names=['Healthy', 'Misaligned'], output_dict=True)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr = fp / (fp + tn)
mcc = matthews_corrcoef(y_test, y_pred)
results['Model'].append('CNN (Ours)')
results['Accuracy'].append(test_acc)
results['AUC'].append(test_auc)
results['Recall'].append(report['Misaligned']['recall'])
results['FPR'].append(fpr)
results['MCC'].append(mcc)
results['Training Time (s)'].append(cnn_time / 5)

# Train and evaluate SVM
X_train_features = extract_time_features(X_train_windows)
X_test_features = extract_time_features(X_test_windows)
svm = SVC(kernel='rbf', probability=True, random_state=42)
start_time = time.time()
svm.fit(X_train_features, y_train)
svm_time = time.time() - start_time
y_pred_svm = svm.predict(X_test_features)
y_pred_prob_svm = svm.predict_proba(X_test_features)[:, 1]
svm_acc = np.mean(y_pred_svm == y_test)
svm_auc = roc_auc_score(y_test, y_pred_prob_svm)
svm_report = classification_report(y_test, y_pred_svm, output_dict=True, target_names=['Healthy', 'Misaligned'])
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_svm).ravel()
fpr = fp / (fp + tn)
mcc = matthews_corrcoef(y_test, y_pred_svm)
results['Model'].append('SVM')
results['Accuracy'].append(svm_acc)
results['AUC'].append(svm_auc)
results['Recall'].append(svm_report['Misaligned']['recall'])
results['FPR'].append(fpr)
results['MCC'].append(mcc)
results['Training Time (s)'].append(svm_time)

# Train and evaluate Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
start_time = time.time()
rf.fit(X_train_features, y_train)
rf_time = time.time() - start_time
y_pred_rf = rf.predict(X_test_features)
y_pred_prob_rf = rf.predict_proba(X_test_features)[:, 1]
rf_acc = np.mean(y_pred_rf == y_test)
rf_auc = roc_auc_score(y_test, y_pred_prob_rf)
rf_report = classification_report(y_test, y_pred_rf, output_dict=True, target_names=['Healthy', 'Misaligned'])
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
fpr = fp / (fp + tn)
mcc = matthews_corrcoef(y_test, y_pred_rf)
results['Model'].append('Random Forest')
results['Accuracy'].append(rf_acc)
results['AUC'].append(rf_auc)
results['Recall'].append(rf_report['Misaligned']['recall'])
results['FPR'].append(fpr)
results['MCC'].append(mcc)
results['Training Time (s)'].append(rf_time)

# Train and evaluate 1D-CNN
start_time = time.time()
model_1d = build_1d_cnn_model()
model_1d.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
history_1d = model_1d.fit(X_train_windows, y_train, epochs=20, batch_size=32, validation_split=0.2,
                          callbacks=[early_stopping, lr_callback], class_weight=class_weight_dict, verbose=0)
cnn1d_time = time.time() - start_time
test_loss_1d, test_acc_1d, test_auc_1d = model_1d.evaluate(X_test_windows, y_test, verbose=0)
y_pred_1d = (model_1d.predict(X_test_windows) > 0.5).astype(int)
report_1d = classification_report(y_test, y_pred_1d, output_dict=True, target_names=['Healthy', 'Misaligned'])
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_1d).ravel()
fpr = fp / (fp + tn)
mcc = matthews_corrcoef(y_test, y_pred_1d)
results['Model'].append('1D-CNN')
results['Accuracy'].append(test_acc_1d)
results['AUC'].append(test_auc_1d)
results['Recall'].append(report_1d['Misaligned']['recall'])
results['FPR'].append(fpr)
results['MCC'].append(mcc)
results['Training Time (s)'].append(cnn1d_time)

# Plotting functions
def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_classification_report(report, title, filename):
    report_df = pd.DataFrame(report).transpose().round(3)
    plt.figure(figsize=(8, 4))
    sns.heatmap(report_df.iloc[:-3, :-1], annot=True, cmap='YlGnBu', cbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_history(history, title_prefix, filename_prefix):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{title_prefix} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title_prefix} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_history.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(models, X_tests, y_test, model_names, filename):
    plt.figure(figsize=(8, 6))
    for (y_pred_prob, name) in zip(models, model_names):
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        auc = roc_auc_score(y_test, y_pred_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison(results, metric, filename):
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Model', y=metric, data=results_df, palette='viridis')
    plt.title(f'Model Comparison: {metric}')
    plt.ylabel(metric)
    plt.ylim(0, 1 if metric != 'Training Time (s)' else None)
    for i, v in enumerate(results_df[metric]):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Generate plots
class_names = ['Healthy', 'Misaligned']
plot_confusion_matrix(y_test, y_pred, class_names, 'CNN Confusion Matrix', 'cnn_confusion_matrix.png')
plot_classification_report(report, 'CNN Classification Report', 'cnn_classification_report.png')
plot_training_history(history, 'CNN Training', 'cnn')
plot_confusion_matrix(y_test, y_pred_svm, class_names, 'SVM Confusion Matrix', 'svm_confusion_matrix.png')
plot_confusion_matrix(y_test, y_pred_rf, class_names, 'Random Forest Confusion Matrix', 'rf_confusion_matrix.png')
plot_confusion_matrix(y_test, y_pred_1d, class_names, '1D-CNN Confusion Matrix', '1d_cnn_confusion_matrix.png')
plot_roc_curves([model.predict(X_test), y_pred_prob_svm, y_pred_prob_rf, model_1d.predict(X_test_windows)],
                [X_test, X_test_features, X_test_features, X_test_windows], y_test,
                ['CNN', 'SVM', 'Random Forest', '1D-CNN'], 'roc_curves.png')
plot_comparison(results, 'Accuracy', 'comparison_accuracy.png')
plot_comparison(results, 'Recall', 'comparison_recall.png')
plot_comparison(results, 'Training Time (s)', 'comparison_time.png')

# SHAP explanation
explainer = shap.DeepExplainer(model, X_train[:50])
shap_values = explainer.shap_values(X_test[:5])
shap.image_plot(shap_values, X_test[:5], show=False)
plt.savefig('shap_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('model_comparison.csv', index=False)
print("\nModel Comparison Results:")
print(results_df.round(3))

# Architecture diagram
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='cnn_architecture.png', show_shapes=True, dpi=300)

# Optimal threshold
fpr, tpr, thresholds = roc_curve(y_test, model.predict(X_test))
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"\nCNN Optimal Threshold: {optimal_threshold:.4f}")