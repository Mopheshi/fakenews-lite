// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        // Remove active class from all tabs
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

        // Add active class to clicked tab
        tab.classList.add('active');
        document.getElementById(tab.dataset.tab + '-tab').classList.add('active');

        // Hide results when switching tabs
        hideResults();
    });
});

// Analyze button click
document.getElementById('analyze-btn').addEventListener('click', analyzeArticle);

// Allow Enter key in URL field
document.getElementById('url').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        analyzeArticle();
    }
});

function analyzeArticle() {
    const activeTab = document.querySelector('.tab.active').dataset.tab;
    const btn = document.getElementById('analyze-btn');

    let payload = {
        input_type: activeTab
    };

    if (activeTab === 'url') {
        const url = document.getElementById('url').value.trim();
        if (!url) {
            showError('Please enter a URL');
            return;
        }
        payload.url = url;
    } else {
        const title = document.getElementById('title').value.trim();
        const text = document.getElementById('text').value.trim();
        if (!text) {
            showError('Please enter the article text');
            return;
        }
        payload.title = title;
        payload.text = text;
    }

    // Show loading state
    btn.disabled = true;
    btn.querySelector('.btn-text').style.display = 'none';
    btn.querySelector('.btn-loading').style.display = 'flex';
    hideResults();

    // Make API request
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showResult(data);
        } else {
            showError(data.error);
        }
    })
    .catch(error => {
        showError('Network error. Please try again.');
        console.error('Error:', error);
    })
    .finally(() => {
        // Reset button state
        btn.disabled = false;
        btn.querySelector('.btn-text').style.display = 'block';
        btn.querySelector('.btn-loading').style.display = 'none';
    });
}

function showResult(data) {
    const resultSection = document.getElementById('result-section');
    const errorSection = document.getElementById('error-section');

    errorSection.style.display = 'none';
    resultSection.style.display = 'block';

    // Set icon and prediction
    const isFake = data.prediction === 'FAKE';
    document.getElementById('result-icon').textContent = isFake ? '🚨' : '✅';

    const predictionEl = document.getElementById('result-prediction');
    predictionEl.textContent = isFake ? 'FAKE NEWS DETECTED' : 'APPEARS LEGITIMATE';
    predictionEl.className = isFake ? 'fake' : 'legit';

    // Set confidence bar
    const confidenceFill = document.getElementById('confidence-fill');
    confidenceFill.style.width = data.confidence + '%';
    confidenceFill.className = 'confidence-fill ' + (isFake ? 'fake' : 'legit');

    document.getElementById('confidence-value').textContent = data.confidence;

    // Set probabilities
    document.getElementById('fake-prob').textContent = data.fake_probability + '%';
    document.getElementById('legit-prob').textContent = data.legit_probability + '%';

    // Show extracted content if from URL
    const extractedContent = document.getElementById('extracted-content');
    if (data.title || data.text_preview) {
        extractedContent.style.display = 'block';
        document.getElementById('extracted-title').textContent = data.title || '(No title extracted)';
        document.getElementById('extracted-text').textContent = data.text_preview || '(No text extracted)';
    } else {
        extractedContent.style.display = 'none';
    }

    // Scroll to result
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showError(message) {
    const resultSection = document.getElementById('result-section');
    const errorSection = document.getElementById('error-section');

    resultSection.style.display = 'none';
    errorSection.style.display = 'block';

    document.getElementById('error-message').textContent = message;
}

function hideResults() {
    document.getElementById('result-section').style.display = 'none';
    document.getElementById('error-section').style.display = 'none';
}

