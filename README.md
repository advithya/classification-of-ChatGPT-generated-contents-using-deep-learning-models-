# chatgpt_classification

### Installation and Running Guide
Prerequisites
Python 3.x
pip (Python package installer)

Step-by-Step Guide
1. Clone the Repository:
If your project is in a repository, clone it using:

   git clone <repository_url>
   cd project_folder

   2. Create a Virtual Environment:
It's a good practice to create a virtual environment to manage dependencies.

   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

   3. Install Dependencies:
Install all the required packages using pip. If you have a requirements.txt file, you can install all dependencies at once
   pip install -r requirements.txt

   If you don't have a requirements.txt file, you can manually install the packages:
      pip install streamlit pandas joblib seaborn scikit-learn nltk gensim matplotlib torch transformers wordcloud flask

      4. Download NLTK Data:
Some NLTK functionalities require additional data. Run the following Python commands to download the necessary datasets:

   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')

   5. Run the Streamlit Application:
Use the Streamlit CLI to run your application.
   streamlit run app.py

 6. Access the Application:
After running the above command, Streamlit will provide a local URL (usually http://localhost:8501). Open this URL in your web browser to access the application.

Summary of Commands

# Clone the repository
git clone <repository_url>
cd project_folder

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Run the Streamlit application
streamlit run app.py

Summary of Commands

# Clone the repository
git clone <repository_url>
cd project_folder

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Run the Streamlit application
streamlit run app.py

Summary of the Folder


Project Overview

This project is focused on classifying ChatGPT-generated content using various machine learning models. The application is built using Streamlit for the web interface and includes functionalities for text preprocessing, model training, and prediction.

Folder Structure

project_folder/
│
├── app.py
├── requirements.txt
├── data/
│   └── dataset.csv
├── models/
│   └── model.pkl
├── static/
│   └── style.css
└── templates/
    └── index.html


Key Files and Directories
1. app.py:
The main application file.
Contains code for loading data, preprocessing text, selecting and training models, and running the Streamlit web app.
Implements various machine learning models like Naive Bayes, KNN, Logistic Regression, SVM, and Decision Tree.

2. requirements.txt:
Lists all the dependencies required to run the project.
Example contents:
     streamlit
     pandas
     joblib
     seaborn
     scikit-learn
     nltk
     gensim
     matplotlib
     torch
     transformers
     wordcloud
     flask

3. data/:
Contains datasets used by the application.
Example: dataset.csv which might be the training data for the models.

4. models/:
Contains pre-trained models.
Example: model.pkl which is a serialized model file.

5. static/:
Contains static files like CSS for styling the web application.
Example: style.css.

6. templates/:
Contains HTML templates if using Flask for additional web functionalities.
Example: index.html.

Main Functionalities


1. Text Preprocessing:
Functions like wordpre(text) and process_text(text) are used to clean and preprocess the text data.

2. Model Loading and Training:
Functions like load_prediction_model(model) and model(clf) are used to load and train machine learning models.

3. User Interface:
Streamlit is used to create an interactive web interface.
Users can input text, select models, and view predictions and metrics.

4. Prediction and Evaluation:
The application allows users to classify text and evaluate the performance of different models.
Metrics like accuracy, precision, recall, and F1-score are computed and displayed.
