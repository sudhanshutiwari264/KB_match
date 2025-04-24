# MaaS360 KB Match Assistant

![MaaS360 KB Match Assistant](https://img.shields.io/badge/MaaS360-KB%20Match%20Assistant-1E88E5)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

A powerful tool for matching support tickets with relevant knowledge base articles using natural language processing techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

The MaaS360 KB Match Assistant is a Streamlit-based web application designed to help support agents quickly find relevant knowledge base articles that match customer issues. By using text similarity algorithms, it identifies the most relevant articles from the knowledge base, saving time and improving customer support efficiency.

## ✨ Features

- **Natural Language Processing**: Uses TF-IDF vectorization to understand the semantic meaning of support tickets
- **Smart Matching**: Ranks KB articles by relevance to the issue description
- **Category Filtering**: Filter results by specific categories (Connectivity, Applications, Security, etc.)
- **Confidence Scoring**: Each match includes a confidence score to indicate relevance
- **Responsive UI**: Clean, intuitive interface that works on desktop and mobile devices
- **Example Queries**: Pre-built examples to demonstrate the tool's capabilities
- **Tier 2 Escalation**: Built-in workflow for escalating unresolved issues
- **Solution Tracking**: Records which KB articles successfully resolved issues

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/kb-match-streamlit-app.git
   cd kb-match-streamlit-app
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Usage

1. Start the Streamlit application:
   ```bash
   streamlit run kb_match_app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Enter a support ticket description in the text area

4. Click "Find KB Articles" to see matching knowledge base articles

5. Review the suggested articles and their relevance scores

6. If an article solves the issue, click "This solved my issue" to record the solution

7. If no articles solve the issue, use the "Escalate to Tier 2 Support" option

## ⚙️ How It Works

The application uses the following process to match support tickets with KB articles:

1. **Text Preprocessing**: The issue description and KB articles are processed to remove common words and normalize text

2. **TF-IDF Vectorization**: The text is converted into numerical vectors using Term Frequency-Inverse Document Frequency (TF-IDF)

3. **Similarity Calculation**: Cosine similarity is used to measure how similar the issue description is to each KB article

4. **Ranking**: Articles are ranked by their similarity score, with the most relevant articles displayed first

5. **Filtering**: Users can filter results by category and minimum confidence threshold

## 🛠️ Customization

### Adding New KB Articles

To add new knowledge base articles, edit the `kb_articles` list in the `kb_match_app.py` file:

```python
kb_articles.append({
    "id": "KB2345",
    "title": "Your Article Title",
    "content": "Detailed content of the knowledge base article...",
    "category": "Your Category"
})
```

### Modifying Categories

Categories are automatically extracted from the KB articles. To add a new category, simply include it in a new or existing KB article.

### Adjusting the UI

The UI can be customized by modifying the CSS in the `st.markdown()` section at the beginning of the application.

### Changing the Matching Algorithm

The current implementation uses TF-IDF vectorization with cosine similarity. For more advanced matching, you could replace this with:

- Sentence transformers (requires PyTorch)
- Word embeddings like Word2Vec or GloVe
- BERT or other transformer-based models

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Built with ❤️ for MaaS360 support teams
