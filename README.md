# YouTube Sentiment Analysis Project

## Overview

This project performs sentiment analysis on YouTube comments using machine learning. The system includes:


- Flask backend API for sentiment processing
- Chrome extension for interactive visualization
- ML pipeline managed with DVC for reproducibility
- MLflow for experiment tracking

## Features

- **Sentiment Analysis**: Classifies YouTube comments into positive, neutral, or negative categories.
- **Trend Analysis**: Displays sentiment trends over time.
- **Word Cloud Generation**: Creates a word cloud from YouTube comments.
- **Chrome Extension**: Provides an interactive interface for analyzing YouTube video comments.
- **Machine Learning Pipeline**: Includes preprocessing, model training, evaluation, and registration using MLflow.
- **DVC Integration**: Tracks data and model versioning.

## Project Structure

```
.
├── data/                      # Data storage (raw, processed)
├── dvc.yaml, dvc.lock        # DVC pipeline files  
├── flask_api/                # Flask API source code
├── notebooks/                # Jupyter notebooks for EDA
├── src/                      # ML model code
├── yt-chrome-plugin-frontend/ # Chrome extension
├── requirements.txt          # Python dependencies
├── setup.py                  # Package configuration
└── params.yaml              # Model parameters
```


## Setup

### Prerequisites

- Python 3.11
- Conda
- DVC
- Chrome browser

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Youtube-Sentiment-analysis
   ```

2. **Create and activate Conda environment:**
   ```bash
   conda create -n youtube python=3.11 -y
   conda activate youtube
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize DVC:**
   ```bash
   dvc init
   dvc repro
   ```

## Usage

### Flask API

1. Start the backend server:
   ```bash
   cd flask_api
   python main.py
   ```

2. API will be available at `http://localhost:5000`

### Chrome Extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked" and select `yt-chrome-plugin-frontend` directory
4. Visit any YouTube video to analyze comments

### Usage and Testing

1. Open a YouTube video.
2. Use the Chrome extension to fetch comments and analyze their sentiment.
3. View insights such as sentiment distribution, trends, and word clouds.

## Key Components

### Backend API
- Located in [flask_api/main.py](flask_api/main.py).
- Provides endpoints for generating charts, word clouds, and sentiment predictions.

### Chrome Extension
- Frontend code is in [yt-chrome-plugin-frontend](yt-chrome-plugin-frontend/).
- Displays sentiment analysis results interactively.

### Machine Learning Pipeline
- Preprocessing, model training, and evaluation scripts are in [src/](src/).
- Uses MLflow for experiment tracking and model registry.


## Development Workflow

1. Data versioning and pipeline steps are managed through DVC
2. Model experiments are tracked in MLflow
3. Frontend changes can be tested locally using the Chrome extension
4. API changes can be tested using the Flask development server

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[MIT License](LICENSE)

## Contact

Author: Aby Philip  
Email: abyphilipaby1996@gmail.com
