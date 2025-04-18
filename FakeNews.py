from google.colab import drive
drive.mount('/content/drive')

!pip install transformers

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,Input
from transformers import BertModel, TFBertModel 
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import numpy as np
import pandas as pd
import os
import time
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from PIL import Image
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.corpus import indian
nltk.download('indian')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
import re
from nltk.stem.porter import PorterStemmer

df = pd.read_csv('/content/drive/MyDrive/datasets/news_dataset.csv')

df = df.dropna()

df.head()

df.info()

label = df['label']
sns.set_style('whitegrid')
sns.countplot(label)

plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.label == 'REAL'].text))
plt.imshow(wc, interpolation = 'bilinear')

plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.label == 'FAKE'].text))
plt.imshow(wc, interpolation = 'bilinear')

df.label = pd.Categorical(df.label)

df['label'] = df.label.cat.codes

df.reset_index(inplace=True)

df = df.drop(['index'],axis=1)

def clean_text(text):
  text = text.lower()
  text = text.strip()
  text = re.sub('\n','',text)
  text = re.sub('\'','',text)
  text = re.sub('((www.[^s]+)|(https?://[^s]+))','',text)
  text = re.sub('|','',text)
  text = re.sub('/','',text)
  text = re.sub('`','',text)
  text = re.sub('"','',text)
  text = re.sub("'",'',text)
  text = re.sub('!','',text)
  text = re.sub(',','',text)
  text = re.sub(':','',text)
  text = re.sub(r"[\([{})\]]", "", text)
  text = re.sub('<','',text)
  text = re.sub('>','',text)
  text = re.sub('-',' ',text)
  text = text.replace('?','')
  text = text.replace('*','')
  text = text.replace('|',' ')
  return text

def remove_stopwords_and_lemmatization(text):
    final_text = []
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    for word in text:
        if word not in (set(stopwords.words('english')) or set(nltk.corpus.indian.words('hindi.pos'))):
            lemma = nltk.WordNetLemmatizer()
            word = lemma.lemmatize(word) 
            final_text.append(word)
    return " ".join(final_text)

def cleaning(text):
  text = clean_text(text)
  text = remove_stopwords_and_lemmatization(text)
  return text

df['text'] = df['text'].apply(cleaning)

max_len=128
data_text=df["text"]
data_label=df["label"]

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')

X_train, X_test, Y_train, Y_test = train_test_split(data_text, data_label, stratify = data_label, test_size = 0.2, random_state = 42,shuffle=True)

def tokenize(X):
    
    X = bert_tokenizer(
        text = list(X),
        add_special_tokens = True,
        max_length = 128,
        truncation = True,
        padding = 'max_length',
        return_tensors = 'np',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True
        )
    return X

X_train_token = tokenize(X_train)
X_test_token = tokenize(X_test)

maxlen = 128

def create_model():
    dropout_rate=0.2
    input_ids=Input(shape=(maxlen,),dtype=tf.int32)
    input_mask=Input(shape=(maxlen,),dtype=tf.int32)
    bert_layer=bert_model([input_ids,input_mask])[1]
    x=Dropout(0.5)(bert_layer)
    x=Dense(64,activation="tanh")(x)
    x=Dropout(0.2)(x)
    x=Dense(1,activation="sigmoid")(x)
    model = Model(inputs=[input_ids, input_mask], outputs=x)
    return model

model=create_model()
model.summary()

tf.keras.utils.plot_model(model)

optimizer = Adam(learning_rate=1e-05, epsilon=1e-08, decay=0.01,clipnorm=1.0)
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = 'accuracy')

history = model.fit(x = {'input_1':X_train_token['input_ids'],'input_2':X_train_token['attention_mask']}, y = Y_train, epochs=5, validation_split = 0.2, batch_size = 32)

import pickle

filename = 'bert.pkl'
pickle.dump(model,open(filename,'wb'))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

y_pred = np.where(model.predict({ 'input_1' : X_test_token['input_ids'] , 'input_2' : X_test_token['attention_mask']}) >=0.5,1,0)

y_pred.shape

accuracy_score(y_pred,Y_test)

from mlxtend.plotting import plot_confusion_matrix
conf_matrix = confusion_matrix(Y_test,y_pred)
fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6))
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

print(classification_report(Y_test,y_pred))

test_text = 'मध्यप्रदेश के कूनो नेशनल पार्क के बड़े बाड़े में छोडे़ जाने के बाद अफ्रीकी चीतों ने खुद पहली बार शिकार किया। दोनों चीतों ने करीब 48 घंटे बाद एक चीतल का शिकार किया। कूनो नेशनल पार्क प्रबंधन ने सोमवार को इसकी जानकारी दी। पार्क प्रबंधन ने बताया कि दोनों चीते स्वस्थ हैं। वे बड़े बाड़े में घूम रहे हैं। डीएफओ प्रकाश वर्मा ने बताया कि कॉलर आईडी, सीसीटीवी और ट्रैप कैमरों से टीम लगातार उन पर नजर रख रही है। अफ्रीका के नामीबिया से 17 सितंबर को 8 चीते कूनो नेशनल पार्क लाए गए थे। PM नरेंद्र मोदी ने इन्हें छोटे बाड़े में छोड़ा था। यहां चीतों को क्वारंटाइन किया गया था। इसके करीब 50 दिन बाद शनिवार को दो नर चीतों को बड़े बाड़े में छोड़ा गया था। दोनों चीतों को बड़े बाड़े में छोड़ने से पहले चीता टास्क फोर्स की बैठक हुई। टास्क फोर्स के सदस्यों की सहमति के बाद चीतों को छोड़ा गया था। खुद पेट भर सकेंगे चीते डीएफओ वर्मा ने बताया कि, लंबे अरसे के बाद देश की धरती पर लाए गए चीतों ने पहली बार शिकार किया है। अब वह खुद पसंदीदा जानवर का शिकार करके खुद अपना पेट भर सकेंगे। छह अन्य चीतों को रिलीज करने का निर्णय चीता टास्क फोर्स करेगा। हमारी तैयारियां पूरी हैं। यह भी पढ़ें कूनो में दो बड़े बाड़े में छोड़ा, पीएम मोदी बोले -ग्रेट न्यूज'

test_token = tokenize(test_text)

test_text_pred = np.where(model.predict({ 'input_1' : test_token['input_ids'] , 'input_2' : test_token['attention_mask']}) >=0.5,1,0)

if(test_text_pred[0]==0):
    print("News is Fake")
else:
    print("News is Real")

























!pip install git+https://github.com/marcotcr/lime.git

import lime
from lime.lime_text import LimeTextExplainer

import torch

class_names = ['True','Fake']

explainer = LimeTextExplainer(class_names=class_names)

test1 = X_test[624]

test1_token = tokenize(test1)

def predict_probab(STR):
    z = bert_tokenizer.encode_plus(STR, add_special_tokens = True, max_length = 128, truncation = True,padding = 'max_length', return_token_type_ids=False, return_attention_mask = True,  return_tensors = 'np')
    inputs = [z['input_ids'], z['attention_mask']]
    k = []
    k.append(float(model.predict(inputs).reshape(-1,1)))
    k.append(float(1-model.predict(inputs).reshape(-1,1)))
    k = np.array(k).reshape(1,-1)
    
    return k

exp = explainer.explain_instance(test1,predict_probab,num_features=100,num_samples=1)

exp.show_in_notebook(text=test1)

