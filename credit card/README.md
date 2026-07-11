# Credit Card Approval Prediction System Using Machine Learning

An AI-powered decision support web application that predicts credit card application approvals or rejections. The system utilizes machine learning ensemble classifiers to assess applicant profiles, calculate default risk metrics, and generate banking-style decision dashboards.

---

## Key Features
- **Parallel Classifier Benchmarking**: Evaluates and compares 11 machine learning algorithms (Logistic Regression, XGBoost, Random Forest, Support Vector Machines, KNN, Naive Bayes, Decision Trees, AdaBoost, Extra Trees, Gradient Boosting, and soft Voting Classifiers).
- **Explainable Decision Reports**: Calculates dynamic credit scores (300-850), AI confidence ratings, risk levels (Low/Medium/High), local feature contribution graphs, and personalized improvement suggestions.
- **Interactive Banking Dashboard**: Rich visual layouts including light/dark theme switches, real-time analytics indicators, search, pagination filters, and printable PDF summary layouts.
- **Bulk Spreadsheet Predictions**: Supports processing CSV spreadsheets, returning predictions, and outputting downloadable evaluation files.
- **Swagger-Style REST APIs**: Exposes secure API integration points (`/api/predict`) with developer consoles for microservice hooks.

---

## Tech Stack
- **Languages**: Python, SQL, JavaScript, HTML5, CSS3
- **Web Layer**: Flask, Jinja2, Bootstrap 5, Chart.js
- **Machine Learning**: Scikit-Learn, XGBoost, Pandas, NumPy, Joblib
- **Database**: SQLite 3
- **Containerization & Hosting**: Docker, Docker Compose, Gunicorn

---

## Installation & Setup

### 1. Clone & Set Up Directory
Navigate to the project root directory.

### 2. Configure Virtual Environment
Create and activate a virtual environment to manage dependencies:
```bash
# Create Environment
python -m venv venv

# Activate Environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate Environment (Mac/Linux Bash)
source venv/bin/activate
```

### 3. Install Package Dependencies
Install scientific packages and server modules:
```bash
pip install -r requirements.txt
```

### 4. Execute ML Pipeline (Train Models)
Run the training pipeline script. This will generate a realistic synthetic dataset (if missing), perform cleaning, train and benchmark all 11 classifiers, output a performance leaderboard, save static exploratory graphs, and serialize the best classifier:
```bash
python train.py
```

### 5. Launch the Web Application
Start the local Flask development server:
```bash
python app.py
```
Open your browser and visit: `http://localhost:5000`

---

## Docker Container Deployment

### Running with Docker Build:
```bash
# Build Docker image
docker build -t credit-card-predictor .

# Run container on port 5000
docker run -p 5000:5000 credit-card-predictor
```

### Running with Docker Compose:
```bash
docker-compose up --build
```
The container will auto-train the models and launch a production-ready Gunicorn server accessible at `http://localhost:5000`.

---

## Project Structure
```
credit-card-approval/
├── dataset/                  # Input CSV records directory
├── models/                   # Serialized preprocessors and model binaries
├── static/                   # Styling, javascript assets, and exploratory graphs
│   ├── css/                  # Custom theme variables and panels layout
│   ├── js/                   # Client-side validation and charts scripts
│   └── images/               # Preloader outputs and static graphics
├── templates/                # Responsive HTML pages
├── tests/                    # Integration and route validators
├── docs/                     # Architecture and schema manuals
├── logs/                     # System logs
├── app.py                    # Flask application controller
├── train.py                  # ML training script
├── predict.py                # Single/Bulk prediction pipeline wrapper
├── config.py                 # Central configurations
├── database.py               # SQLite schema controller
├── requirements.txt          # Python package requirements
├── Dockerfile                # Image build configuration
└── README.md                 # Project user manual
```
