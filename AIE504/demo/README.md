# Fake News Detector - Web Demo

A clean, minimalistic web interface for testing the fake news detection model.

## Features

- 📝 **Text Input**: Paste article title and text directly
- 🔗 **URL Input**: Automatically extract content from news URLs
- 🎯 **Real-time Analysis**: Get instant predictions with confidence scores
- 📊 **Detailed Results**: View fake/legit probabilities

## Setup

### 1. Install Dependencies

```bash
pip install flask requests beautifulsoup4
```

### 2. Train the Model

Before running the web app, you need to train and save the model:

```bash
cd demo
python train_model.py
```

This will:
- Load the WELFake dataset
- Train an XGBoost model
- Save it to `saved_model.pkl`

### 3. Run the Web App

```bash
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

## Usage

### Option 1: Paste Text
1. Click the "📝 Paste Text" tab
2. Enter the article headline (optional)
3. Paste the article content
4. Click "Analyze Article"

### Option 2: From URL
1. Click the "🔗 From URL" tab
2. Enter the news article URL
3. Click "Analyze Article"
4. The app will automatically extract and analyze the content

## Project Structure

```
demo/
├── app.py              # Flask application
├── train_model.py      # Model training script
├── saved_model.pkl     # Saved model (after training)
├── templates/
│   └── index.html      # HTML template
├── static/
│   ├── style.css       # Styles
│   └── script.js       # JavaScript
└── README.md           # This file
```

## API Endpoints

- `GET /` - Main page
- `POST /analyze` - Analyze article (JSON payload)
- `GET /health` - Health check

## Notes

- The model uses XGBoost with TF-IDF features
- URL extraction works best with standard news article pages
- Some websites may block automated content extraction

