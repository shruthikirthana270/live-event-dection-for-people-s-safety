from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog

import numpy as np
import pandas as pd
import seaborn as sns
import os
import pickle
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import librosa
import librosa.display
import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier #load ML classes
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

accuracy = []
precision = []
recall = []
fscore = []

model_folder = "model"

def remove_noise(audio, sr):
    noise_profile = audio[:int(0.5 * sr)]
    noise_reduced_audio = librosa.effects.remix(audio, intervals=librosa.effects.split(audio, top_db=20))
    return noise_reduced_audio


def load_audio_with_features(path, categories, model_folder):
    X_file = os.path.join(model_folder, "X.npy")
    Y_file = os.path.join(model_folder, "Y.npy")

    if os.path.exists(X_file) and os.path.exists(Y_file):
        print(f"Loading cached data from {model_folder}")
        X = np.load(X_file)
        Y = np.load(Y_file)
    else:
        print(f"Path does not exist: {path}" if not os.path.exists(path) else "Processing directory")
        X = []
        Y = []

        for root, dirs, files in os.walk(path):
            print(f"Processing root: {root}")
            for file in files:
                name = os.path.basename(root)
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    try:
                        y, sr = librosa.load(file_path, sr=None)  # Load the audio file

                        # Extracting 10 different features
                        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30).T, axis=0)
                        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
                        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
                        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
                        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
                        rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
                        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0)
                        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
                        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)
                        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)

                        # Concatenating features into a single vector
                        feature_vector = np.hstack([
                            mfccs, chroma, mel, spectral_contrast, tonnetz,
                            rms, spectral_bandwidth, spectral_centroid, spectral_rolloff, zero_crossing_rate
                        ])

                        X.append(feature_vector)
                        if name in categories:
                            Y.append(categories.index(name))
                        else:
                            print(f"Category {name} not in categories list.")
                    except Exception as e:
                        print(f"Skipping {file_path}, error reading file: {e}")

        X = np.array(X)
        Y = np.array(Y)
        os.makedirs(model_folder, exist_ok=True)  # Ensure the directory exists
        np.save(X_file, X)
        np.save(Y_file, Y)

    return X, Y

def Upload_Dataset():
    global filename,categories
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    categories = [d for d in os.listdir(filename) if os.path.isdir(os.path.join(filename, d))]
    text.insert(END,'Dataset loaded\n')
    text.insert(END,"Classes found in dataset: "+str(categories)+"\n")




def Preprocess_Dataset():
    global X,Y,filename,categories,df

    path=filename
    X, Y = load_audio_with_features(path, categories, model_folder)
    # Normalize features
    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)
    text.insert(END, "Preprocessing and MFCC Feature Extraction completed on Dataset: " + str(filename) + "\n\n")
    text.insert(END, "Input MFCC Feature Set Size: " + str(X.shape) + "\n\n")
    
    X_list = X.tolist()
    Y_list = Y.tolist()
    df = pd.DataFrame({'X': X_list, 'Y': Y_list})
    
    # Plot the counts of each category
    category_counts = {category: len(os.listdir(os.path.join(path, category))) for category in categories}
    df_counts = pd.DataFrame(list(category_counts.items()), columns=['Category', 'Count'])
    plt.figure(figsize=(10, 6))
    plt.bar(df_counts['Category'], df_counts['Count'], color='skyblue')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Number of Sounds per Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    


def Train_Test_Splitting():
    global X,Y
    global x_train,y_train,x_test,y_test

    # Create a count plot


    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    
# Display information about the dataset
    text.delete('1.0', END)
    text.insert(END, "Total records found in dataset: " + str(X.shape[0]) + "\n\n")
    text.insert(END, "Total records found in dataset to train: " + str(x_train.shape[0]) + "\n\n")
    text.insert(END, "Total records found in dataset to test: " + str(x_test.shape[0]) + "\n\n")

def Calculate_Metrics(algorithm, predict, y_test):
    global categories

    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    conf_matrix = confusion_matrix(y_test, predict)
    
    CR = classification_report(y_test, predict,target_names=categories)
    text.insert(END,algorithm+' Classification Report \n')
    text.insert(END,algorithm+ str(CR) +"\n\n")

    
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = categories, yticklabels = categories, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(categories)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()       

def existing_classifier():
    
    global x_train,y_train,x_test,y_test,mlpc1
    text.delete('1.0', END)

    path = 'model/MLPClassifier'
    if(os.path.exists(path)):
        mlpc1 = joblib.load(path) 
    else:
        mlpc1 = MLPClassifier()
        mlpc1.fit(x_train, y_train)
        joblib.dump(mlpc1,path)
    y_pred = mlpc1.predict(x_test)
    Calculate_Metrics("Existing MLP Classifier", y_pred, y_test)


def existing_classifier1():
    
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)
    
    path = 'model/KNN'
    if(os.path.exists(path)):
        mlpc = joblib.load(path) 
    else:
        mlpc = KNeighborsClassifier()
        mlpc.fit(x_train, y_train)
        joblib.dump(mlpc,path)
    y_pred = mlpc.predict(x_test)
    Calculate_Metrics("Existing KNN Classifier", y_pred, y_test)

    

def existing_classifier2():
    
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)

    path = 'model/DTC'
    if(os.path.exists(path)):
        mlpc = joblib.load(path) 
    else:
        mlpc = DecisionTreeClassifier()
        mlpc.fit(x_train, y_train)
        joblib.dump(mlpc,path)
    y_pred = mlpc.predict(x_test)
    Calculate_Metrics("Existing DTC", y_pred, y_test)

    

def existing_classifier3():
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)

    path = 'model/LRC'
    if(os.path.exists(path)):
        mlpc = joblib.load(path) 
    else:
        mlpc = LogisticRegression(C=0.01, penalty='l1',solver='liblinear')
        mlpc.fit(x_train, y_train)
        joblib.dump(mlpc,path)
    y_pred = mlpc.predict(x_test)
    Calculate_Metrics("Existing LRC", y_pred, y_test)
    

        

def existing_classifier4():
    global x_train,y_train,x_test,y_test
    text.delete('1.0', END)

    path = 'model/LRC'
    if(os.path.exists(path)):
        mlpc = joblib.load(path) 
    else:
        mlpc = AdaBoostClassifier()
        mlpc.fit(x_train, y_train)
        joblib.dump(mlpc,path)
    y_pred = mlpc.predict(x_test)
    Calculate_Metrics("Existing AdaBoost Classifier", y_pred, y_test)



    
def existing_classifier5():
    global df,Y,X_train, X_test, y_train, y_test
    
    df_resampled = resample(df, replace=True, n_samples=4000, random_state=42)
    X_new = df_resampled['X'].tolist()
    Y_new = df_resampled['Y'].tolist()
    
    smote = SMOTE(random_state=42)
    X_resampled, Y_resampled = smote.fit_resample(X_new, Y_new)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, Y_resampled, test_size = 0.20, random_state = 44)


    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    unique_before, counts_before = np.unique(Y, return_counts=True)
    sns.barplot(x=unique_before, y=counts_before)
    plt.title('Class Distribution Before Data Balancing')
    for index, value in enumerate(counts_before):
        plt.text(index, value, str(value), ha='center', va='bottom')
    
    # Plot countplot after data balancing
    plt.subplot(1, 2, 2)
    unique_after, counts_after = np.unique(Y_resampled, return_counts=True)
    sns.barplot(x=unique_after, y=counts_after)
    plt.title('Class Distribution After Data Balancing')
    for index, value in enumerate(counts_after):
        plt.text(index, value, str(value), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def proposed_classifier():
    global X_train, X_test, y_train, y_test,lgbm
    text.delete('1.0', END)

    path = 'model/LGBMClassifier'
    if(os.path.exists(path)):
        lgbm = joblib.load(path) 
    else:
        lgbm = LGBMClassifier()
        lgbm.fit(X_train, y_train)
        joblib.dump(lgbm,path)
    y_pred = lgbm.predict(X_test)
    Calculate_Metrics("Proposed Light Gradiant Boosting Classifier", y_pred, y_test)

# Prediction on new test data
def preprocess_audio(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=None)

        # Extracting 10 different features
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=30).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr).T, axis=0)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr).T, axis=0)
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr).T, axis=0)
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)

        # Concatenating features into a single vector
        feature_vector = np.hstack([
            mfccs, chroma, mel, spectral_contrast, tonnetz,
            rms, spectral_bandwidth, spectral_centroid, spectral_rolloff, zero_crossing_rate
        ])

        return feature_vector

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None
def Prediction():
    global mlpc1, categories

    filename = filedialog.askopenfilename(initialdir="Test Data")
    text.delete('1.0', END)
    text.insert(END, f'{filename} Loaded\n')

    X_new = preprocess_audio(filename)
    X_new = X_new.reshape(1, -1)

    prediction = mlpc1.predict(X_new)
    predicted_category = categories[prediction[0]]

    text.insert(END, f"Predicted Outcome From Test Audio is: {predicted_category}\n\n")

    # Load audio file
    y, sr = librosa.load(filename, sr=None)

    # Create a waveplot
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Save plot to an image buffer
    plt.savefig("waveplot.png")
    plt.close()

    # Load the saved waveplot image
    waveplot_img = cv2.imread("waveplot.png")
    if waveplot_img is not None:
        # Add the predicted category text on the waveplot
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, 50)  # Position to place the text
        font_scale = 1
        font_color = (0, 0, 255)  # Red color in BGR
        thickness = 2
        line_type = cv2.LINE_AA

        cv2.putText(waveplot_img, f"Predicted: {predicted_category}", position, font, font_scale, font_color, thickness, line_type)

        # Display the waveplot with annotation
        cv2.imshow("Waveplot with Prediction", waveplot_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

def close():
    main.destroy()


main = tkinter.Tk()
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")

font = ('times', 18, 'bold')
title = Label(main, text="Live Event Detection for People’s Safety Using NLP and Light GBM")
title.config(bg='white', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
Button1 = Button(main, text="Upload Dataset", command=Upload_Dataset)
Button1.place(x=20,y=100)
Button1.config(font=font1)

Button1 = Button(main, text="Preprocess Dataset", command=Preprocess_Dataset)
Button1.place(x=20,y=150)
Button1.config(font=font1)

Button1 = Button(main, text="Train Test Splitting", command=Train_Test_Splitting)
Button1.place(x=20,y=200)
Button1.config(font=font1) 


Button1 = Button(main, text="Existing MLP", command=existing_classifier)
Button1.place(x=20,y=250)
Button1.config(font=font1)

Button1 = Button(main, text="K Nearest Neighbour", command=existing_classifier1)
Button1.place(x=20,y=300)
Button1.config(font=font1)

Button1 = Button(main, text="Decision Tree Classifier", command=existing_classifier2)
Button1.place(x=20,y=350)
Button1.config(font=font1)


Button1 = Button(main, text="Logistic Regression Classifier", command=existing_classifier3)
Button1.place(x=20,y=400)
Button1.config(font=font1)

Button1 = Button(main, text="AdaBoost Classifier", command=existing_classifier4)
Button1.place(x=20,y=450)
Button1.config(font=font1)

Button1 = Button(main, text="SMOTE Data Balancing", command=existing_classifier5)
Button1.place(x=20,y=450)
Button1.config(font=font1)

Button1 = Button(main, text="Proposed LGBM", command=proposed_classifier)
Button1.place(x=20,y=500)
Button1.config(font=font1)

Button1 = Button(main, text="Prediction", command=Prediction)
Button1.place(x=20,y=550)
Button1.config(font=font1)


Button1 = Button(main, text="Exit", command=close)
Button1.place(x=20,y=600)
Button1.config(font=font1)


                            
font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=95)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=300,y=100)
text.config(font=font1)
main.config(bg='SeaGreen1')

main.mainloop()