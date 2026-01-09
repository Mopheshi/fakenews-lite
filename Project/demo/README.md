# Fake News Detector - Lightweight Ensemble Model with Web Demo

A high-performance fake news detection system combining XGBoost and TextCNN models with part-of-speech features.
Achieves **98.09% accuracy** with minimal computational requirements.

## 🎯 Key Features

- 🤖 **Ensemble Learning**: Combines XGBoost (statistical) and TextCNN (deep learning) through soft voting
- 📊 **Advanced Features**: TF-IDF with SVD dimensionality reduction + Part-of-Speech linguistic features
- 🎓 **Pre-trained Embeddings**: Uses frozen GloVe 100d embeddings for efficient semantic understanding
- 📝 **Dual Input Modes**: Direct text input or automatic URL content extraction
- ⚡ **Real-time Analysis**: Get instant predictions with confidence scores (<2 seconds)
- 💻 **CPU-Only**: No GPU required - runs efficiently on standard hardware
- 📈 **High Accuracy**: 98.09% accuracy, 0.9977 ROC-AUC on WELFake dataset

## 📊 Model Performance

| Model        | Accuracy   | Precision  | Recall     | F1-Score   | ROC-AUC    |
|--------------|------------|------------|------------|------------|------------|
| XGBoost      | 95.75%     | 94.95%     | 96.89%     | 95.91%     | 0.9925     |
| TextCNN      | 97.90%     | 97.89%     | 98.03%     | 97.96%     | 0.9974     |
| **Ensemble** | **98.09%** | **97.93%** | **98.37%** | **98.15%** | **0.9977** |

## 🚀 Quick Start

### Prerequisites

```bash
# Core dependencies
pip install pandas numpy scikit-learn xgboost nltk

# Deep learning
pip install tensorflow keras

# Web application
pip install flask requests beautifulsoup4

# Optional: For faster processing
pip install lxml
```

### Download Required Resources

```bash
# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### Dataset and Embeddings

1. **WELFake Dataset**: Place `WELFake_Dataset.csv` in `../files/`
2. **GloVe Embeddings** (optional but recommended): Download `glove.6B.100d.txt`
   from [Stanford NLP](https://nlp.stanford.edu/projects/glove/) and place in `../files/`

## 📦 Installation & Training

### Step 1: Train the Ensemble Model

```bash
cd demo
python train_model.py
```

**What happens during training:**

1. **Data Loading**: Loads 72,134 news articles from WELFake dataset
    - Discovered label correction: Label 0 = Legit, Label 1 = Fake (opposite of documentation)

2. **Preprocessing** (~5 minutes):
    - Text cleaning (HTML, URLs, special characters removal)
    - Lemmatization and stopword removal
    - POS feature extraction (nouns, verbs, adjectives, adverbs, pronouns, modals)
    - Optimised batch processing with caching

3. **XGBoost Training** (~5 minutes):
    - TF-IDF vectorization (10,000 features, 1-3 n-grams)
    - SVD dimensionality reduction (300 components)
    - Combined with 6 POS features → 306-dimensional input
    - Achieves 95.75% accuracy

4. **TextCNN Training** (~15 minutes):
    - Tokenization with 50,000 vocabulary size
    - 500-token sequences with padding
    - GloVe embedding initialisation (frozen for efficiency)
    - 3 parallel Conv1D branches (kernel sizes: 3, 4, 5)
    - Early stopping at epoch 4
    - Achieves 97.90% accuracy

5. **Ensemble Creation**:
    - Soft voting combines both model probabilities
    - Final accuracy: **98.09%**

6. **Model Saving**:
    - Saves to `saved_model.pkl` (includes XGBoost, TextCNN, preprocessor, feature extractors, and ensemble)

**Expected output:**

```
TRAINING OPTIMISED ENSEMBLE MODEL FOR WEB DEMO
============================================================

LOADING DATA...
  Dataset loaded successfully
  Shape: (72134, 4)
  Label distribution:
    Legit news (0): 35028
    Fake news (1): 37106

[... preprocessing and training logs ...]

MODEL COMPARISON
============================================================
                      accuracy  precision    recall        f1   roc_auc
XGBoost               0.957504   0.949538  0.968864  0.959103  0.992493
TextCNN               0.978995   0.978869  0.980321  0.979595  0.997412
Ensemble_soft_voting  0.980936   0.979334  0.983691  0.981508  0.997729

Model saved successfully!
You can now run the web app with: python app.py
```

### Step 2: Run the Web Application

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## 🖥️ Usage Guide

### Option 1: Paste Text

1. Click the **"📝 Paste Text"** tab
2. Enter the article headline (optional but recommended)
3. Paste the article content
4. Click **"Analyse Article"**
5. View results with confidence scores and probability distributions

### Option 2: From URL

1. Click the **"🔗 From URL"** tab
2. Enter the complete news article URL (include https://)
3. Click **"Analyse Article"**
4. The app automatically extracts title and content
5. View results with extracted content preview

### Understanding Results

**Prediction Labels:**

- ✅ **APPEARS LEGITIMATE**: Article classified as real news
- 🚨 **FAKE NEWS DETECTED**: Article classified as fake news

**Confidence Score**: Overall confidence in the prediction (0-100%)

**Probability Distribution**:

- **Fake Probability**: Likelihood the article is fake
- **Legit Probability**: Likelihood the article is legitimate

**Example Output:**

```
✅ APPEARS LEGITIMATE
95.9% confidence

Fake Probability:  4.1%
Legit Probability: 95.9%
```

## 📂 Project Structure

```
demo/
├── app.py                      # Flask web application
├── train_model.py              # Complete ensemble training pipeline
├── saved_model.pkl             # Trained ensemble model (generated after training)
│
├── templates/
│   └── index.html              # Main web interface
│
├── static/
│   ├── style.css               # Styling and animations
│   └── script.js               # Client-side interaction logic
│
└── README.md                   # This file

../modules/                     # Core ML modules (parent directory)
├── data_loader.py              # Dataset loading utilities
├── text_processor.py           # Text preprocessing with POS extraction
├── feature_extractor.py        # TF-IDF, SVD, tokenization, GloVe loading
├── fake_news_detector.py       # Unified prediction interface
└── model_evaluator.py          # Performance metrics and visualization

../models/                      # Model architectures (parent directory)
├── xgboost_model.py            # XGBoost classifier
├── text_cnn_model.py           # TextCNN with GloVe embeddings
└── ensemble_model.py           # Soft voting ensemble

../files/                       # Data files (parent directory)
├── WELFake_Dataset.csv         # Training dataset (72,134 articles)
└── glove.6B.100d.txt           # Pre-trained word embeddings (optional)
```

## 🔧 Configuration Options

Edit `train_model.py` to customise:

```python
# Enable/disable POS features
USE_POS_FEATURES = True  # Adds linguistic features

# Train both models or just one
TRAIN_BOTH_MODELS = True  # Set False to train only XGBoost

# Ensemble method
ENSEMBLE_METHOD = 'soft_voting'  # Options: 'soft_voting', 'weighted_voting', 'stacking'
```

**Ensemble Methods:**

- `soft_voting`: Average probabilities (default, most balanced)
- `weighted_voting`: F1-score weighted average
- `stacking`: Meta-learner (Logistic Regression) on base predictions

## 🌐 API Endpoints

### `GET /`

Main web interface

### `POST /analyse`

Analyse article for fake news

**Request Body (JSON):**

```json
{
  "input_type": "text",
  "title": "Article headline",
  "text": "Article content..."
}
```

OR

```json
{
  "input_type": "url",
  "url": "https://example.com/article"
}
```

**Response (JSON):**

```json
{
  "success": true,
  "prediction": "LEGIT",
  "confidence": 95.9,
  "fake_probability": 4.1,
  "legit_probability": 95.9,
  "title": "Article headline...",
  "text_preview": "Article content preview...",
  "model_type": "Ensemble"
}
```

### `GET /health`

Health check endpoint

**Response:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "Ensemble"
}
```

## ⚙️ Technical Details

### XGBoost Model Configuration

```python
var = {
    'max_depth': 7,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'min_child_weight': 3,
    'gamma': 0.1,
    'subsample': 0.9,
    'colsample_bytree': 0.9
}
```

### TextCNN Architecture

- **Embedding Layer**: 50,000 vocab × 100 dim (GloVe, frozen)
- **Conv1D Branches**: 3 parallel (kernel sizes: 3, 4, 5, each with 128 filters)
- **Global Max Pooling**: Extract most important features
- **Dense Layer**: 128 units with ReLU
- **Dropout**: 0.5 rate
- **Output**: Sigmoid activation for binary classification

### POS Features Extracted

1. **Nouns** (NN, NNS, NNP, NNPS)
2. **Verbs** (VB, VBD, VBG, VBN, VBP, VBZ)
3. **Adjectives** (JJ, JJR, JJS)
4. **Adverbs** (RB, RBR, RBS)
5. **Pronouns** (PRP, PRP$, WP, WP$)
6. **Modal verbs** (MD)

## 🐛 Troubleshooting

### Model Not Found Error

```
NO SAVED MODEL FOUND!
Please run the training script first:
  python train_model.py
```

**Solution**: Train the model first before running the web app.

### NLTK Data Missing

```
LookupError: Resource punkt not found
```

**Solution**: Download NLTK data:

```python
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### URL Extraction Fails

```
Error fetching URL: Connection refused
```

**Possible causes:**

- Website blocks automated requests (use User-Agent headers)
- Website requires JavaScript (our extractor uses static HTML)
- Invalid/inaccessible URL

**Solution**: Use text input mode instead, or try a different news source.

### Out of Memory During Training

**Solution**: Reduce batch size or vocabulary size in `train_model.py`:

```python
# In FeatureExtractor
max_words = 30000  # Reduce from 50000
max_len = 300  # Reduce from 500
```

## 📝 Important Notes

### Dataset Label Correction

⚠️ **Critical Discovery**: The WELFake dataset documentation states "Label 0 = fake and 1 = real", but our analysis
revealed the **actual encoding is reversed**:

- **Label 0 = Legitimate/Real News**
- **Label 1 = Fake News**

This was verified through manual inspection of article content. All code and evaluations use the corrected labels.

### URL Extraction Limitations

- Works best with standard news article pages
- Some websites block automated access
- JavaScript-heavy sites may not extract properly
- Paywalled content cannot be accessed

### Model Limitations

- Trained on English-language news only
- May not generalise to other languages or cultures
- Performance may degrade on AI-generated text
- Does not detect satire, opinion, or nuanced misinformation

## 🔬 For Researchers

### Reproducing Results

```bash
# Use the exact configuration in train_model.py
python train_model.py

# Results should match:
# - XGBoost: 95.75% accuracy
# - TextCNN: 97.90% accuracy  
# - Ensemble: 98.09% accuracy
```

### Data Split

- Training: 70% (50,487 samples)
- Validation: 10% (7,213 samples)
- Test: 20% (14,425 samples)
- Stratified splitting maintains class balance

### Citation

If you use this work in your research, please cite:

```
@article{Edward2025lightweight,
  title={Lightweight Machine Learning Models for Fake News Detection: An Ensemble Approach with Part-of-Speech Features},
  author={Edward, Ndachimya Magaji},
  journal={AIE501 Seminar Term Project},
  year={2025},
  institution={Istanbul Okan University}
}
```

## 📊 Performance Benchmarks

**Hardware:**

- CPU: Standard desktop/laptop processor
- RAM: 8GB minimum, 16GB recommended
- Storage: ~2GB for models and data

**Training Time:**

- Preprocessing: ~5 minutes
- XGBoost: ~5 minutes
- TextCNN: ~15 minutes
- **Total: ~25 minutes** (CPU-only, no GPU)

**Inference Time:**

- Average: <2 seconds per article
- Web app response: <3 seconds (including network)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Multilingual support
- Additional ensemble methods
- Real-time monitoring dashboard
- Browser extension
- Mobile application
- Adversarial robustness testing

## 📄 License

This project is for academic and research purposes. The WELFake dataset follows its original licence terms.

## 🙏 Acknowledgments

- **WELFake Dataset**: Verma et al. (2021)
- **GloVe Embeddings**: Stanford NLP
- **XGBoost**: Chen & Guestrin (2016)
- **TextCNN**: Kim (2014)