"""
Fake News Detection Web Application
A simple Flask-based web UI for testing fake news detection models.
"""

import os
import pickle
import re
import sys
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup

# Import project modules
from modules.fake_news_detector import FakeNewsDetector

app = Flask(__name__)

# Global variables for models
detector = None
preprocessor = None
feature_extractor = None
model = None
model_loaded = False


def extract_article_from_url(url):
    """Extract article title and text from a URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Try to extract title
        title = ""
        if soup.find('h1'):
            title = soup.find('h1').get_text(strip=True)
        elif soup.find('title'):
            title = soup.find('title').get_text(strip=True)

        # Try to extract article text
        text = ""

        # Common article containers
        article_selectors = [
            'article',
            '[class*="article"]',
            '[class*="content"]',
            '[class*="post"]',
            '[class*="story"]',
            'main',
            '.entry-content',
            '#content'
        ]

        for selector in article_selectors:
            article = soup.select_one(selector)
            if article:
                # Get all paragraphs
                paragraphs = article.find_all('p')
                if paragraphs:
                    text = ' '.join([p.get_text(strip=True) for p in paragraphs])
                    if len(text) > 200:  # Reasonable article length
                        break

        # Fallback: get all paragraphs from page
        if len(text) < 200:
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text(strip=True) for p in paragraphs])

        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()

        if not title and not text:
            return None, None, "Could not extract content from this URL"

        return title, text, None

    except requests.exceptions.RequestException as e:
        return None, None, f"Error fetching URL: {str(e)}"
    except Exception as e:
        return None, None, f"Error parsing content: {str(e)}"


def load_models():
    """Load or train the models"""
    global detector, preprocessor, feature_extractor, model, model_loaded

    model_path = os.path.join(os.path.dirname(__file__), 'saved_model.pkl')

    if os.path.exists(model_path):
        print("Loading saved model...")
        try:
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)

            preprocessor = saved_data['preprocessor']
            feature_extractor = saved_data['feature_extractor']
            model = saved_data['model']

            detector = FakeNewsDetector(
                model=model,
                preprocessor=preprocessor,
                feature_extractor=feature_extractor,
                model_type='xgboost',
                use_pos_features=False
            )
            model_loaded = True
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    else:
        print("=" * 60)
        print("NO SAVED MODEL FOUND!")
        print("=" * 60)
        print("\nPlease run the training script first to create a model:")
        print("  python train_model.py")
        print("\nOr run the main notebook to train the models.")
        print("=" * 60)
        return False


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', model_loaded=model_loaded)


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyse text or URL for fake news"""
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        })

    data = request.json
    input_type = data.get('input_type', 'text')

    title = ""
    text = ""

    if input_type == 'url':
        url = data.get('url', '').strip()
        if not url:
            return jsonify({'success': False, 'error': 'Please provide a URL'})

        title, text, error = extract_article_from_url(url)
        if error:
            return jsonify({'success': False, 'error': error})

    else:  # text input
        title = data.get('title', '').strip()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'success': False, 'error': 'Please provide article text'})

    try:
        # Make prediction
        result = detector.predict_article(title, text, verbose=False)

        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': round(result['confidence'] * 100, 1),
            'fake_probability': round(result['fake_probability'] * 100, 1),
            'legit_probability': round(result['legit_probability'] * 100, 1),
            'title': title[:200] + '...' if len(title) > 200 else title,
            'text_preview': text[:500] + '...' if len(text) > 500 else text
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f'Analysis error: {str(e)}'})


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("FAKE NEWS DETECTION - WEB DEMO")
    print("=" * 60 + "\n")

    if load_models():
        print("\nStarting web server...")
        print("Open http://127.0.0.1:5000 in your browser\n")
        app.run(debug=True, host='127.0.0.1', port=5000)
    else:
        print("\nCannot start server without a trained model.")
        print("Please run 'python train_model.py' first.")
