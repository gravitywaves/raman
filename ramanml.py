import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, roc_curve, auc, precision_recall_curve, 
                           average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import datetime  # Add this import at the beginning of your script
import warnings
warnings.filterwarnings('ignore')


def visualize_spectra(normal_spectra, tumor_spectra, wavenumbers):
    """Visualize average spectra and their differences"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot average spectra
    normal_mean = np.mean(normal_spectra, axis=0)
    normal_std = np.std(normal_spectra, axis=0)
    tumor_mean = np.mean(tumor_spectra, axis=0)
    tumor_std = np.std(tumor_spectra, axis=0)
    
    # Average spectra with standard deviation
    ax1.plot(wavenumbers, normal_mean, 'b-', label='Normal Tissue', alpha=0.7)
    ax1.fill_between(wavenumbers, normal_mean - normal_std, normal_mean + normal_std, 
                     color='b', alpha=0.2)
    ax1.plot(wavenumbers, tumor_mean, 'r-', label='Tumor Tissue', alpha=0.7)
    ax1.fill_between(wavenumbers, tumor_mean - tumor_std, tumor_mean + tumor_std, 
                     color='r', alpha=0.2)
    ax1.set_title('Average Raman Spectra with Standard Deviation')
    ax1.set_xlabel('Wavenumber (cm⁻¹)')
    ax1.set_ylabel('Intensity (a.u.)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Difference spectrum
    diff_spectrum = tumor_mean - normal_mean
    ax2.plot(wavenumbers, diff_spectrum, 'g-', label='Difference (Tumor - Normal)')
    ax2.set_title('Difference Spectrum')
    ax2.set_xlabel('Wavenumber (cm⁻¹)')
    ax2.set_ylabel('Difference in Intensity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Heatmap of all spectra
    combined_spectra = np.vstack((normal_spectra, tumor_spectra))
    sns.heatmap(combined_spectra, cmap='viridis', ax=ax3)
    ax3.set_title('Heatmap of All Spectra')
    ax3.set_xlabel('Wavenumber Index')
    ax3.set_ylabel('Sample Index (Normal -> Tumor)')
    
    plt.tight_layout()
    plt.show()

def analyze_feature_importance(X_train, y_train, X_val, y_val, wavenumbers):
    """Analyze and visualize feature importance using multiple methods"""
    # Train Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_importance = rf.feature_importances_
    
    # Train XGBoost for feature importance
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_importance = xgb_model.feature_importances_
    
    # Plot feature importances
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Random Forest importance
    ax1.plot(wavenumbers, rf_importance)
    ax1.set_title('Random Forest Feature Importance')
    ax1.set_xlabel('Wavenumber (cm⁻¹)')
    ax1.set_ylabel('Importance')
    ax1.grid(True, alpha=0.3)
    
    # XGBoost importance
    ax2.plot(wavenumbers, xgb_importance)
    ax2.set_title('XGBoost Feature Importance')
    ax2.set_xlabel('Wavenumber (cm⁻¹)')
    ax2.set_ylabel('Importance')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Return top important features
    top_features = pd.DataFrame({
        'Wavenumber': wavenumbers,
        'RF_Importance': rf_importance,
        'XGB_Importance': xgb_importance
    }).sort_values(by='RF_Importance', ascending=False).head(10)
    
    return top_features

def evaluate_model_enhanced(y_true, y_pred, y_prob=None):
    """Enhanced evaluation metrics"""
    basic_metrics = {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'precision': precision_score(y_true, y_pred) * 100,
        'recall': recall_score(y_true, y_pred) * 100,
        'f1_score': f1_score(y_true, y_pred) * 100
    }
    
    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    basic_metrics.update({
        'specificity': (tn / (tn + fp)) * 100,
        'sensitivity': (tp / (tp + fn)) * 100,
        'npv': (tn / (tn + fn)) * 100,  # Negative Predictive Value
        'ppv': (tp / (tp + fp)) * 100   # Positive Predictive Value
    })
    
    if y_prob is not None:
        # ROC and PR curve metrics
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        basic_metrics.update({
            'auc_roc': auc(fpr, tpr),
            'auc_pr': average_precision_score(y_true, y_prob),
            'roc_curve': (fpr, tpr),
            'pr_curve': (precision, recall)
        })
    
    return basic_metrics

def load_and_preprocess_data(wavenumbers_file, normal_file, tumor_part1_file, tumor_part2_file):
    """Load and preprocess the Raman spectral data from CSV files"""
    # Load wavenumbers
    wavenumbers = pd.read_csv(wavenumbers_file)
    
    # Load spectral data
    normal_data = pd.read_csv(normal_file)
    tumor_data1 = pd.read_csv(tumor_part1_file)
    # Load tumor_data2 from the file 
    tumor_data2 = pd.read_csv(tumor_part2_file) # This line is changed!
    
    # Combine tumor data
    tumor_data = pd.concat([tumor_data1, tumor_data2], axis=1) # This line is changed!
    
    # Convert to numpy arrays and remove any non-numeric columns
    normal_spectra = normal_data.select_dtypes(include=[np.number]).values.T
    tumor_spectra = tumor_data.select_dtypes(include=[np.number]).values.T
    
    print("Data shapes:")
    print(f"Normal spectra: {normal_spectra.shape}")
    print(f"Tumor spectra: {tumor_spectra.shape}")
    print(f"Wavenumbers: {wavenumbers.shape}")
    
    return normal_spectra, tumor_spectra, wavenumbers

# Original CNN architecture from human_1to24.py
class HumanConv1D(nn.Module):
    def __init__(self):
        super(HumanConv1D, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm1d(num_features=32, affine=False),
            nn.ReLU(),
        )
        self.avgpool = nn.AvgPool1d(3, stride=1)
        self.cls = nn.Sequential(nn.Linear(6976, 1, bias=False))

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.conv(x)
        x1 = x1.view(x1.size(0), -1)
        x = self.cls(x1)
        return x

# Enhanced CNN architecture
class EnhancedCNN1D(nn.Module):
    def __init__(self, input_size):
        super(EnhancedCNN1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(8)
        )
        
        # Residual connections
        self.skip1 = nn.Conv1d(1, 32, 1)
        self.skip2 = nn.Conv1d(32, 64, 1)
        self.skip3 = nn.Conv1d(64, 128, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        # Residual connections
        skip1 = self.skip1(x)
        x = self.conv1(x)
        x = x + skip1[:, :, :x.shape[2]]
        
        skip2 = self.skip2(x)
        x = self.conv2(x)
        x = x + skip2[:, :, :x.shape[2]]
        
        skip3 = self.skip3(x)
        x = self.conv3(x)
        x = x + skip3[:, :, :x.shape[2]]
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class LSTM1D(nn.Module):
    def __init__(self, input_size):
        super(LSTM1D, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=128, num_layers=2, 
                           batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# Enhanced LSTM with attention mechanism
class AttentionLSTM1D(nn.Module):
    def __init__(self, input_size):
        super(AttentionLSTM1D, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=128, num_layers=2,
                           batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def attention_net(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context

    def forward(self, x):
        x = x.permute(0, 2, 1)
        lstm_output, _ = self.lstm(x)
        attention_output = self.attention_net(lstm_output)
        output = self.fc(attention_output)
        return output

def evaluate_model(y_true, y_pred):
    """Calculate various metrics for model evaluation"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp)
    sens = tp / (tp + fn)
    return {
        'accuracy': acc * 100,
        'precision': prec * 100,
        'recall': rec * 100,
        'f1_score': f1 * 100,
        'specificity': spec * 100,
        'sensitivity': sens * 100
    }

def balance_dataset(X, y, method='smote'):
    """Balance dataset using specified method"""
    print(f"\nOriginal dataset shape: {Counter(y)}")
    
    if method == 'smote':
        balancer = SMOTE(random_state=42)
    elif method == 'undersample':
        balancer = RandomUnderSampler(random_state=42)
    elif method == 'hybrid':
        rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
        X_temp, y_temp = rus.fit_resample(X, y)
        balancer = SMOTE(random_state=42)
        X_balanced, y_balanced = balancer.fit_resample(X_temp, y_temp)
        print(f"Balanced dataset shape: {Counter(y_balanced)}")
        return X_balanced, y_balanced
    
    X_balanced, y_balanced = balancer.fit_resample(X, y)
    print(f"Balanced dataset shape: {Counter(y_balanced)}")
    return X_balanced, y_balanced

def train_neural_net(model, train_loader, val_loader, device, class_weights=None, epochs=70):
    """Train neural network models"""
    # Move class_weights to the correct device if provided
    if class_weights is not None:
        class_weights = class_weights.to(device)  
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights) if class_weights is not None else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0
    best_metrics = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            # Move batch_x and batch_y to the correct device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device) 
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # ... (rest of the function remains the same)
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                preds = torch.sigmoid(outputs) > 0.5
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())
        
        val_metrics = evaluate_model(val_true, val_preds)
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_metrics = val_metrics
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss/len(train_loader):.4f}')
            print(f'Val Accuracy: {val_metrics["accuracy"]:.2f}%')
    
    return best_metrics

def visualize_model_comparisons(all_results):
    """Visualize performance comparisons between models"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'sensitivity']
    balance_methods = list(all_results.keys())
    models = list(all_results[balance_methods[0]].keys())
    
    # Create subplots for each metric
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Prepare data for plotting
        data = []
        labels = []
        for method in balance_methods:
            metric_values = [all_results[method][model][metric] 
                           for model in models if metric in all_results[method][model]]
            data.append(metric_values)
            labels.extend([f"{model}\n({method})" for model in models])
        
        # Create boxplot
        bp = ax.boxplot(data, labels=balance_methods, patch_artist=True)
        
        # Customize colors
        colors = ['lightblue', 'lightgreen', 'lightpink']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_ylabel('Percentage')
        ax.grid(True, alpha=0.3)
        ax.set_xticklabels(balance_methods, rotation=45)
    
    plt.tight_layout()
    plt.show()

def visualize_learning_curves(model, train_loader, val_loader, device, epochs=70):
    """Visualize learning curves during training"""
    train_losses = []
    train_accs = []
    val_accs = []
    val_losses = []
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        epoch_train_correct = 0
        epoch_train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            epoch_train_correct += (preds == batch_y).sum().item()
            epoch_train_total += batch_y.size(0)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_correct = 0
        epoch_val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                epoch_val_loss += loss.item()
                preds = torch.sigmoid(outputs) > 0.5
                epoch_val_correct += (preds == batch_y).sum().item()
                epoch_val_total += batch_y.size(0)
        
        # Record metrics
        train_losses.append(epoch_train_loss / len(train_loader))
        train_accs.append(100 * epoch_train_correct / epoch_train_total)
        val_losses.append(epoch_val_loss / len(val_loader))
        val_accs.append(100 * epoch_val_correct / epoch_val_total)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.2f}%')
            print(f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.2f}%')
    
    # Plot learning curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return train_losses, train_accs, val_losses, val_accs

def analyze_architectures():
    """Analyze and compare CNN architectures"""
    # Original CNN Architecture Analysis
    original_params = sum(p.numel() for p in HumanConv1D().parameters())
    
    print("Original CNN Architecture Analysis:")
    print("-" * 50)
    print("Layer Structure:")
    print("1. Input Layer: 1 channel")
    print("2. Convolutional Layer:")
    print("   - Input channels: 1")
    print("   - Output channels: 32")
    print("   - Kernel size: 5")
    print("   - Stride: 2")
    print("   - No bias")
    print("3. Batch Normalization (non-affine)")
    print("4. ReLU Activation")
    print("5. Average Pooling:")
    print("   - Kernel size: 3")
    print("   - Stride: 1")
    print("6. Fully Connected Layer:")
    print("   - Input: 6976")
    print("   - Output: 1")
    print(f"Total Parameters: {original_params:,}")
    print("\nKey Characteristics:")
    print("- Simple architecture with single conv layer")
    print("- No dropout (relies on batch norm for regularization)")
    print("- Xavier weight initialization")
    print("- Average pooling instead of max pooling")
    
    print("\n" + "=" * 50 + "\n")
    
    # Enhanced CNN Architecture Analysis
    enhanced_params = sum(p.numel() for p in EnhancedCNN1D(440).parameters())
    
    print("Enhanced CNN Architecture Analysis:")
    print("-" * 50)
    print("Layer Structure:")
    print("1. Input Layer: 1 channel")
    print("2. First Convolutional Block:")
    print("   - Conv1d: 1 → 32 channels, kernel=7, stride=2")
    print("   - Batch Normalization")
    print("   - ReLU")
    print("   - MaxPool1d")
    print("3. Second Convolutional Block:")
    print("   - Conv1d: 32 → 64 channels, kernel=5")
    print("   - Batch Normalization")
    print("   - ReLU")
    print("   - MaxPool1d")
    print("4. Third Convolutional Block:")
    print("   - Conv1d: 64 → 128 channels, kernel=3")
    print("   - Batch Normalization")
    print("   - ReLU")
    print("   - AdaptiveMaxPool1d")
    print("5. Residual Connections:")
    print("   - Skip connections between conv blocks")
    print("6. Fully Connected Layers:")
    print("   - 128*8 → 256 → 128 → 1")
    print("   - Dropout layers (0.5, 0.3)")
    print(f"Total Parameters: {enhanced_params:,}")
    print("\nKey Characteristics:")
    print("- Deeper architecture with 3 conv layers")
    print("- Residual connections for better gradient flow")
    print("- Multiple dropout layers for regularization")
    print("- Adaptive pooling for input size flexibility")
    print("- Gradually increasing channel depth")
    
    # Architecture Comparison
    print("\nArchitecture Comparison:")
    print("-" * 50)
    print(f"Parameter Ratio (Enhanced/Original): {enhanced_params/original_params:.2f}x")
    print("\nKey Differences:")
    print("1. Depth: Enhanced is deeper with 3 conv layers vs 1")
    print("2. Regularization: Enhanced uses dropout, Original uses only batch norm")
    print("3. Pooling: Enhanced uses max pooling, Original uses average pooling")
    print("4. Skip Connections: Enhanced has residual connections, Original is linear")
    print("5. FC Layers: Enhanced has multiple FC layers with dropout")
    print("6. Parameter Count: Enhanced has more parameters for better feature extraction")
def visualize_holdout_comparison(train_results, val_results, test_results, model_name):
    """Visualize performance comparison across train, validation, and test sets"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'sensitivity']
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.25
    
    plt.bar(x - width, [train_results[m] for m in metrics], width, label='Train')
    plt.bar(x, [val_results[m] for m in metrics], width, label='Validation')
    plt.bar(x + width, [test_results[m] for m in metrics], width, label='Test')
    
    plt.xlabel('Metrics')
    plt.ylabel('Performance (%)')
    plt.title(f'Performance Comparison - {model_name}')
    plt.xticks(x, metrics, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def train_neural_net_holdout(model, train_loader, val_loader, test_loader, device, 
                           class_weights=None, epochs=70):
    """Train neural network with holdout evaluation"""
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights) if class_weights is not None else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0
    best_val_metrics = None
    best_test_metrics = None
    best_train_metrics = None
    best_model_state = None
    
    train_losses = []
    val_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_true = []
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            train_preds.extend(preds.cpu().numpy())
            train_true.extend(batch_y.cpu().numpy())
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        val_probs = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = probs > 0.5
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
        
        train_metrics = evaluate_model_enhanced(train_true, train_preds)
        val_metrics = evaluate_model_enhanced(val_true, val_preds, val_probs)
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        # Evaluate on test set if validation accuracy improves
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_val_metrics = val_metrics
            best_train_metrics = train_metrics
            best_model_state = model.state_dict().copy()
            
            # Test set evaluation
            test_preds = []
            test_true = []
            test_probs = []
            test_loss = 0
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item()
                    
                    probs = torch.sigmoid(outputs)
                    preds = probs > 0.5
                    test_preds.extend(preds.cpu().numpy())
                    test_true.extend(batch_y.cpu().numpy())
                    test_probs.extend(probs.cpu().numpy())
            
            best_test_metrics = evaluate_model_enhanced(test_true, test_preds, test_probs)
            test_losses.append(test_loss / len(test_loader))
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss/len(train_loader):.4f}')
            print(f'Val Loss: {val_loss/len(val_loader):.4f}')
            print(f'Train Accuracy: {train_metrics["accuracy"]:.2f}%')
            print(f'Val Accuracy: {val_metrics["accuracy"]:.2f}%')
    
    # Plot learning curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    if test_losses:
        plt.plot(test_losses, label='Test Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Restore best model
    model.load_state_dict(best_model_state)
    return best_train_metrics, best_val_metrics, best_test_metrics

def compare_methods_holdout(normal_spectra, tumor_spectra, test_size=0.1, val_size=0.15, 
                          balance_method='smote'):
    """Compare different classification methods with smaller holdout testing
    
    Parameters:
    -----------
    test_size : float, default=0.1
        Percentage of data to use for final testing (10%)
    val_size : float, default=0.15
        Percentage of remaining data to use for validation (15% of 90%)
    """
    # Combine data
    X = np.concatenate((normal_spectra, tumor_spectra), axis=0)
    y = np.concatenate((np.zeros(len(normal_spectra)), np.ones(len(tumor_spectra))))
    
    # First split: separate holdout test set (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, 
                                                     random_state=42, stratify=y)
    
    # Second split: separate training and validation sets (15% of remaining data)
    val_size_adj = val_size / (1 - test_size)  # Adjust validation size
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, 
                                                     test_size=val_size_adj, 
                                                     random_state=42, stratify=y_temp)
    
    # Print split sizes for verification
    print(f"\nDataset splits:")
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Balance training data
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train, method=balance_method)
    
    # Visualize data
    visualize_spectra(normal_spectra, tumor_spectra, np.arange(X.shape[1]))
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Analyze feature importance
    importance_results = analyze_feature_importance(X_train_scaled, y_train_balanced, 
                                                 X_val_scaled, y_val, 
                                                 np.arange(X.shape[1]))
    print("\nTop Important Features:")
    print(importance_results)
    
    print(f"\nDataset splits:")
    print(f"Training set: {X_train_scaled.shape[0]} samples")
    print(f"Validation set: {X_val_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    
    # Calculate class weights
    class_weights = torch.tensor([len(y_train[y_train == 0]) / len(y_train[y_train == 1])])
    
    # Prepare PyTorch data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create datasets and loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled).unsqueeze(1),
        torch.FloatTensor(y_train_balanced).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled).unsqueeze(1),
        torch.FloatTensor(y_val).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled).unsqueeze(1),
        torch.FloatTensor(y_test).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    results = {}
    
    # Train and evaluate traditional ML models
    models = {
        'SVM': SVC(kernel='rbf', class_weight='balanced', probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'XGBoost': xgb.XGBClassifier(scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
                                    use_label_encoder=False, eval_metric='logloss'),
        'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000)
    }
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train_balanced)
        
        # Get predictions and probabilities for all sets
        train_preds = model.predict(X_train_scaled)
        val_preds = model.predict(X_val_scaled)
        test_preds = model.predict(X_test_scaled)
        
        train_probs = model.predict_proba(X_train_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        val_probs = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        test_probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Evaluate on all sets
        train_metrics = evaluate_model_enhanced(y_train_balanced, train_preds, train_probs)
        val_metrics = evaluate_model_enhanced(y_val, val_preds, val_probs)
        test_metrics = evaluate_model_enhanced(y_test, test_preds, test_probs)
        
        results[name] = {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
        
        # Visualize performance comparison
        visualize_holdout_comparison(train_metrics, val_metrics, test_metrics, name)
    
    # Analyze and train neural networks
    analyze_architectures()
    
    print("\nTraining Original 1D CNN...")
    original_cnn = HumanConv1D().to(device)
    original_train, original_val, original_test = train_neural_net_holdout(
        original_cnn, train_loader, val_loader, test_loader, device, class_weights
    )
    results['Original CNN'] = {
        'train': original_train,
        'validation': original_val,
        'test': original_test
    }
    visualize_holdout_comparison(original_train, original_val, original_test, 'Original CNN')
    
    print("\nTraining Enhanced 1D CNN...")
    enhanced_cnn = EnhancedCNN1D(X_train_scaled.shape[1]).to(device)
    enhanced_train, enhanced_val, enhanced_test = train_neural_net_holdout(
        enhanced_cnn, train_loader, val_loader, test_loader, device, class_weights
    )
    results['Enhanced CNN'] = {
        'train': enhanced_train,
        'validation': enhanced_val,
        'test': enhanced_test
    }
    visualize_holdout_comparison(enhanced_train, enhanced_val, enhanced_test, 'Enhanced CNN')
    
    return results

if __name__ == '__main__':
    # File paths
    wavenumbers_file = 'x_axis_wavenumbers 1.xlsx.csv'
    normal_file = 'y_normalize_data_NormalTissue_update (2).xlsx.csv'
    tumor_part1_file = 'y_normalize_data_Tumor_Part1_update (2).xlsx.csv'
    tumor_part2_file = 'y_normalize_data_Tumor_Part2_update.xlsx.csv'
    
    # Load and preprocess data
    normal_spectra, tumor_spectra, wavenumbers = load_and_preprocess_data(
        wavenumbers_file, normal_file, tumor_part1_file, tumor_part2_file
    )
    
    # Run comparison with different balancing methods
    balance_methods = ['smote', 'undersample', 'hybrid']
    all_results = {}
    
    for method in balance_methods:
        print(f"\nRunning comparison with {method.upper()} balancing:")
        results = compare_methods_holdout(normal_spectra, tumor_spectra, 
                                        test_size=0.2, val_size=0.2,
                                        balance_method=method)
        all_results[method] = results
    
    # Print comprehensive results
    print("\nFinal Results Summary:")
    print("=" * 120)
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'sensitivity', 'auc_roc']
    sets = ['train', 'validation', 'test']
    
    for balance_method, results in all_results.items():
        print(f"\n{balance_method.upper()} Balancing Results:")
        print("-" * 120)
        
        for set_name in sets:
            print(f"\n{set_name.upper()} Set Results:")
            print("-" * 100)
            
            # Print header
            print(f"{'Method':<15}", end="")
            for metric in metrics:
                print(f"{metric:<12}", end="")
            print("\n" + "-" * 100)
            
            # Print results for each model
            for method, scores in results.items():
                print(f"{method:<15}", end="")
                for metric in metrics:
                    if metric in scores[set_name]:
                        print(f"{scores[set_name][metric]:>11.2f}", end=" ")
                    else:
                        print(f"{'N/A':>11}", end=" ")
                print()
    
    # Plot comparative analysis
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each metric comparing train/val/test performance
    plot_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'sensitivity']  # Remove auc_roc from plotting
    
    for idx, metric in enumerate(plot_metrics):
        plt.subplot(2, 3, idx + 1)
        
        x_pos = np.arange(len(results.keys()))
        width = 0.25
        
        for method_idx, balance_method in enumerate(balance_methods):
            try:
                train_scores = []
                val_scores = []
                test_scores = []
                
                for model in results.keys():
                    # Safely get scores with error handling
                    try:
                        train_scores.append(all_results[balance_method][model]['train'][metric])
                        val_scores.append(all_results[balance_method][model]['validation'][metric])
                        test_scores.append(all_results[balance_method][model]['test'][metric])
                    except (KeyError, TypeError):
                        train_scores.append(np.nan)
                        val_scores.append(np.nan)
                        test_scores.append(np.nan)
                
                # Plot only if we have valid scores
                if any(not np.isnan(score) for score in train_scores):
                    plt.bar(x_pos + method_idx*width - width, train_scores, width, 
                           label=f'{balance_method}-train' if idx == 0 else "", alpha=0.7)
                if any(not np.isnan(score) for score in val_scores):
                    plt.bar(x_pos + method_idx*width, val_scores, width, 
                           label=f'{balance_method}-val' if idx == 0 else "", alpha=0.7)
                if any(not np.isnan(score) for score in test_scores):
                    plt.bar(x_pos + method_idx*width + width, test_scores, width, 
                           label=f'{balance_method}-test' if idx == 0 else "", alpha=0.7)
                    
            except Exception as e:
                print(f"Error plotting {metric} for {balance_method}: {str(e)}")
                continue
        
        plt.title(f'{metric}')
        plt.xticks(x_pos + width, list(results.keys()), rotation=45)
        if idx == 0:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)  # Set y-axis limit for percentage metrics
    
    plt.tight_layout()
    plt.show()
    
    # Save results to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'raman_classification_results_{timestamp}.txt'
    
    with open(results_file, 'w') as f:
        f.write("Raman Classification Results\n")
        f.write("=" * 100 + "\n\n")
        
        for balance_method, results in all_results.items():
            f.write(f"\n{balance_method.upper()} Balancing Results:\n")
            f.write("-" * 100 + "\n")
            
            for set_name in sets:
                f.write(f"\n{set_name.upper()} Set Results:\n")
                f.write("-" * 80 + "\n")
                
                # Write header
                f.write(f"{'Method':<15}")
                for metric in plot_metrics:  # Use plot_metrics instead of metrics
                    f.write(f"{metric:<12}")
                f.write("\n" + "-" * 80 + "\n")
                
                # Write results for each model
                for method, model_results in results.items():
                    f.write(f"{method:<15}")
                    for metric in plot_metrics:  # Use plot_metrics instead of metrics
                        try:
                            if metric in model_results[set_name]:
                                f.write(f"{model_results[set_name][metric]:>11.2f} ")
                            else:
                                f.write(f"{'N/A':>11} ")
                        except (KeyError, TypeError):
                            f.write(f"{'N/A':>11} ")
                    f.write("\n")
                
                f.write("\n")
    
    print(f"\nResults have been saved to {results_file}")
