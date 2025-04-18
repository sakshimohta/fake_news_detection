# fake_news_detection
📰 Fake News Detection Using BERT & Deep Learning
This project leverages BERT embeddings and a deep learning model built using TensorFlow and Keras to classify news articles as real or fake. It uses a combination of NLP techniques, TF-IDF, and transfer learning to boost classification accuracy.

📌 Key Features
🔍 Preprocessing: Cleans text, removes stopwords, and generates word clouds

📊 EDA: Uses Matplotlib and Seaborn for insightful visualizations

🧠 Modeling:

Leverages BERT (Bidirectional Encoder Representations from Transformers) for contextual embeddings

Builds a custom neural network classifier with dense and dropout layers

🎯 Evaluation: Outputs accuracy, confusion matrix, and classification report

📈 Visualization: Displays training performance and data distributions

🧾 Requirements
Install all dependencies using:

bash
Copy
Edit
pip install -r requirements.txt
Or install manually:

bash
Copy
Edit
pip install tensorflow transformers sklearn pandas numpy matplotlib seaborn nltk wordcloud pillow
📂 Project Structure
bash
Copy
Edit
FakeNews/
├── FakeNews.py          # Main script for training and evaluation
├── dataset/             # Folder containing training and test datasets
├── README.md            # Project overview and instructions
└── output/              # (Optional) Saved models, plots, and reports
💻 How to Run
Clone this repo and navigate to the project directory.

Add your dataset (CSV format with text and label columns) to the dataset/ folder.

Run the script:

bash
Copy
Edit
python FakeNews.py
You can also run it in Google Colab for GPU acceleration.

📊 Dataset Requirements
Your dataset should include:


Column	Description
text	News article text
label	0 = Real, 1 = Fake
📌 Sample Output
Confusion Matrix

Accuracy Score

Word Cloud of Most Frequent Words

Training Loss & Accuracy Graphs

✍️ Author
Created by Your Name — feel free to connect or contribute!

