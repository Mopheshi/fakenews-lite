import os
import pickle
import re
import sys
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup

# Import project modules
from modules.fake_news_detector import FakeNewsDetector

app = Flask(__name__)

# Global variables
detector = None
model_loaded = False


def extract_article_from_url(url):
    """Extract article title and text from a URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract title
        title = ""
        if soup.find('h1'):
            title = soup.find('h1').get_text(strip=True)
        elif soup.find('title'):
            title = soup.find('title').get_text(strip=True)

        # Extract text
        text = ""
        article_selectors = [
            'article', '[class*="article"]', '[class*="content"]',
            '[class*="post"]', '[class*="story"]', 'main',
            '.entry-content', '#content'
        ]

        for selector in article_selectors:
            article = soup.select_one(selector)
            if article:
                paragraphs = article.find_all('p')
                if paragraphs:
                    text = ' '.join([p.get_text(strip=True) for p in paragraphs])
                    if len(text) > 200:
                        break

        if len(text) < 200:
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text(strip=True) for p in paragraphs])

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
    global detector, model_loaded

    model_path = os.path.join(os.path.dirname(__file__), 'saved_model.pkl')

    if os.path.exists(model_path):
        print("Loading saved model...")
        try:
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)

            # Extract components
            xgboost_model = saved_data.get('xgboost_model')
            textcnn_model = saved_data.get('textcnn_model')
            preprocessor = saved_data['preprocessor']
            feature_extractor_xgb = saved_data.get('feature_extractor_xgb')
            feature_extractor_cnn = saved_data.get('feature_extractor_cnn')
            ensemble_model = saved_data.get('ensemble_model')
            use_pos_features = saved_data.get('use_pos_features', False)

            # Create ensemble detector
            detector = FakeNewsDetector(
                xgboost_model=xgboost_model,
                textcnn_model=textcnn_model,
                preprocessor=preprocessor,
                feature_extractor_xgb=feature_extractor_xgb,
                feature_extractor_cnn=feature_extractor_cnn,
                ensemble_model=ensemble_model,
                use_pos_features=use_pos_features
            )

            model_loaded = True
            model_type = "Ensemble" if ensemble_model else "Single Model"
            print(f"{model_type} loaded successfully!")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    else:
        print("=" * 60)
        print("NO SAVED MODEL FOUND!")
        print("=" * 60)
        print("\nPlease run the training script first:")
        print("  python train_model.py")
        print("=" * 60)
        return False


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', model_loaded=model_loaded)


@app.route('/analyse', methods=['POST'])
def analyse():
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
            'text_preview': text[:500] + '...' if len(text) > 500 else text,
            'model_type': 'Ensemble' if detector.ensemble_model else 'Single Model'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f'Analysis error: {str(e)}'})


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded,
        'model_type': 'Ensemble' if (detector and detector.ensemble_model) else 'Single Model'
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
        print("Please run 'python train_model_ensemble.py' first.")
