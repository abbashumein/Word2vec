# Word2vec
A CS224N-inspired implementation comparing traditional and embedding-based approaches

🎭 Amazon Sentiment Analysis: TF-IDF vs Word2Vec
A CS224N-inspired implementation comparing traditional and embedding-based approaches
📖 Project Overview
This project implements sentiment analysis on Amazon product reviews, comparing TF-IDF (frequency-based) with Word2Vec (embedding-based) approaches. Built as a practical exploration of Stanford CS224N concepts (Lectures 1 & 2), it demonstrates the strengths and trade-offs between different text vectorization methods.
🎯 Key Results: TF-IDF achieved 89.69% accuracy vs Word2Vec's 87.46%, revealing that frequency-based methods can outperform embeddings for sentiment tasks despite Word2Vec's superior semantic understanding.
🚀 Features

📊 Interactive Web App - Gradio interface for real-time sentiment prediction
⚖️ Model Comparison - Side-by-side TF-IDF vs Word2Vec analysis with confidence scores
🧠 Word2Vec Explorer - Discover semantic similarities and word relationships
📈 Comprehensive Evaluation - Detailed performance metrics and error analysis
🎓 Educational Content - CS224N concepts explained through practical examples

🛠️ Technical Implementation
Models Trained

TF-IDF + Logistic Regression (Baseline): 5000 features, n-grams(1,2)
Word2Vec + Logistic Regression: 100D vectors, CBOW, window=5
Random Forest + Word2Vec: Alternative classifier comparison

Dataset

Source: Amazon Product Reviews (Kaggle)
Size: ~21,000 reviews after cleaning
Classes: Negative (68%), Positive (28%), Neutral (4%)
Challenge: Severe class imbalance handled with SMOTE oversampling

Key Technologies

ML: scikit-learn, Gensim Word2Vec
Interface: Gradio web app
Processing: pandas, numpy, NLTK
Visualization: matplotlib, classification reports

📁 Project Structure
sentiment-analysis/
├── data/
│   ├── raw_reviews.csv           # Original Amazon reviews dataset
│   └── cleaned_reviews.csv       # Preprocessed and labeled data
├── models/
│   ├── tfidf_model.pkl          # Trained TF-IDF vectorizer + classifier
│   ├── word2vec_model.pkl       # Custom Word2Vec embeddings
│   └── model_comparison.ipynb   # Training and evaluation notebook
├── app/
│   ├── gradio_app.py           # Interactive web application
│   ├── model_utils.py          # Helper functions for predictions
│   └── demo_examples.py        # Test cases and examples
├── analysis/
│   ├── results_analysis.ipynb  # Performance comparison and insights
│   └── cs224n_exploration.ipynb # Word vector analysis and analogies
└── README.md
🔬 Key Findings & Insights
Performance Results
ModelAccuracyNegative F1Positive F1Neutral F1TF-IDF + LR89.69%0.940.860.00Word2Vec + LR87.46%0.920.820.06Word2Vec + RF85.85%0.900.800.03
Critical Insights

🏆 TF-IDF Victory: Frequency patterns more predictive than semantic relationships for sentiment
🧠 Word2Vec Strengths: Superior semantic understanding, handles synonyms better
⚡ Efficiency Trade-off: 100D Word2Vec vectors vs 5000D TF-IDF features
😐 Neutral Challenge: Both models struggle with ambiguous neutral sentiment (4% of data)

🚀 Quick Start
Installation
bashgit clone https://github.com/yourusername/sentiment-analysis-comparison
cd sentiment-analysis-comparison
pip install -r requirements.txt
Run the Interactive App
pythonpython app/gradio_app.py
Train Models from Scratch
python# Load and preprocess data
python data/preprocess_reviews.py

# Train models
python models/train_models.py

# Launch comparison interface  
python app/gradio_app.py
🧪 Try These Examples
Test in the app to see model differences:

Clear Sentiment: "This product is absolutely amazing!"
Double Negative: "Not bad at all, actually quite good!"
Sarcasm: "Oh great, another broken product!"
Mixed Signals: "Fast shipping but terrible quality"
Ambiguous: "It's okay, nothing special"

🎓 CS224N Learning Connections
This project demonstrates core concepts from Stanford CS224N:

Lecture 1: Word vectors as numerical representations for NLP tasks
Lecture 2: Word2Vec implementation (CBOW), distributional semantics, vector space models
Practical Applications: Real-world comparison of embedding approaches vs traditional methods

Word2Vec Exploration
Try these in the app's Word2Vec Explorer:

good → excellent, great, nice, fine
terrible → awful, horrible, bad, worst
shipping → delivery, package, service

📊 Future Improvements

 Skip-gram Implementation: Compare CBOW vs Skip-gram architectures
 Pre-trained Embeddings: Integrate GloVe or FastText vectors
 Neural Classifiers: Deep learning models with embedding layers
 Attention Mechanisms: Implement basic attention for document classification
 BERT Comparison: Modern transformer-based sentiment analysis

🤝 Contributing

Fork the repository
Create feature branch (git checkout -b feature/improvement)
Commit changes (git commit -am 'Add new feature')
Push to branch (git push origin feature/improvement)
Open Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE.md file for details.
🙏 Acknowledgments

Stanford CS224N: Natural Language Processing with Deep Learning course
Kaggle: Amazon Product Reviews dataset
Gensim: Word2Vec implementation
Gradio: Interactive ML interface framework
