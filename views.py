from django.shortcuts import render,HttpResponse


from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import login ,logout,authenticate
from django.core.files.storage import default_storage
import tempfile

# Create your views here.

def home(request):
    return render(request,'Home.html')

def register(request):
    if request.method == 'POST':
        First_Name = request.POST['name']
        Email = request.POST['email']
        username = request.POST['username']
        password = request.POST['password']
        confirmation_password = request.POST['cnfm_password']
        select_user=request.POST['role']
        if select_user=='admin':
            admin=True
        else:
            admin=False
        if password == confirmation_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists, please choose a different one.')
                return redirect('register')
            else:
                if User.objects.filter(email=Email).exists():
                    messages.error(request, 'Email already exists, please choose a different one.')
                    return redirect('register')
                else:
                    user = User.objects.create_user(
                        username=username,
                        password=password,
                        email=Email,
                        first_name=First_Name,
                        is_staff=admin
                    )
                    user.save()
                    return redirect('login')
        else:
            messages.error(request, 'Passwords do not match.')
        return render(request, 'register.html')
    return render(request, 'register.html')

def login_view(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        if User.objects.filter(username=username).exists():
            user=User.objects.get(username=username)
            if user.check_password(password):
                user = authenticate(username=username,password=password)
                if user is not None:
                    login(request,user)
                    messages.success(request,'login successfull')
                    return redirect('/')
                else:
                   messages.error(request,'please check the Password Properly')
                   return redirect('login')
            else:
                messages.error(request,"please check the Password Properly")  
                return redirect('login') 
        else:
            messages.error(request,"username doesn't exist")
            return redirect('login')
    return render(request,'login.html')
# Load and preprocess the dataset
def logout_view(request):
    logout(request)
    return redirect('login')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
# Ignore all warnings
warnings.filterwarnings('ignore')
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC,SVR
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split


global X_train,X_test,y_train,y_test
X_train = None
def Upload_data(request):
    load=True
    global X_train,X_test,y_train,y_test
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        df=pd.read_csv(default_storage.path(file_path))
        t=df.keys()
        t
        unique_counts = []
        for col in t:  # You can add more column names here if needed
            unique_count = df[col].nunique()
            unique_counts.append((col, unique_count))
            print(col,"=" ,unique_count)
        # output variable is value
        y=df['RUL']
        X = df.drop(columns=['RUL'])
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=43)
        default_storage.delete(file_path)
        print('---done---')
        outdata=df.head(100)
        return render(request,'prediction.html',{'predict':outdata.to_html()})
    return render(request,'prediction.html',{'upload':load})
mae_list = []
mse_list = []
rmse_list = []
r2_list = []

def calculateMetrics(algorithm,predict, testY):
    
        # Regression metrics
        mae = mean_absolute_error(testY, predict)
        mse = mean_squared_error(testY, predict)
        rmse = np.sqrt(mse)
        r2 = r2_score(testY, predict)
        
        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)
        
        print(f"{algorithm} Mean Absolute Error (MAE): {mae:.2f}")
        print(f"{algorithm} Mean Squared Error (MSE): {mse:.2f}")
        print(f"{algorithm} Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"{algorithm} R-squared (R²): {r2:.2f}")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=testY, y=predict, alpha=0.6)
        plt.plot([min(testY), max(testY)], [min(testY), max(testY)], 'r--', lw=2)  # Line of equality
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(algorithm)
        plt.grid(True)
        plt.show()

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

def dnn(request):
    if X_train is None:
        messages.error(request,'Please upload dataset first')
        return redirect('upload')
    else:
    # Check if the model file exists
        if os.path.exists('static/Model/dnn_model.h5'):
            # Load the trained model
            model = load_model('dnn_model.h5')
            print("Model loaded successfully.")
        else:
            # Build a new model
            model = Sequential([
                Dense(64, input_dim=X_train.shape[1], activation='relu'),
                Dense(32, activation='relu'),
                Dense(16, activation='relu'),
                Dense(1, activation='linear')  # Linear activation for regression
            ])

            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

            # Train the model
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32, callbacks=[early_stopping])

            # Save the model to a file
            model.save('static/Model/dnn_model.h5')
            print("Model saved successfully.")
        # Predict on test data
        y_pred = model.predict(X_test).flatten()
        # Evaluate the model
        calculateMetrics("DNN Model", y_pred, y_test)
    return render(request,'prediction.html',
                  {'algorithm':'DNN Model',
                   'mse_list':mse_list[-1],
                   'r2_list':r2_list[-1],
                   'mae_list':mae_list[-1]})

def SVr(request):
    # Check if the model file exists
    if X_train is None:
        messages.error(request,'Please upload dataset first')
        return redirect('upload')
    else:
        if os.path.exists('static/Model/svr_model.pkl'):
            # Load the trained model
            svr = joblib.load('static/Model/svr_model.pkl')
            print("Model loaded successfully.")
        else:
            # Train a new Support Vector Regressor model
            svr = SVR(kernel='rbf', C=100, epsilon=0.1)
            svr.fit(X_train, y_train)

            # Save the model
            joblib.dump(svr, 'static/Model/svr_model.pkl')
            print("Model saved successfully.")
        # Predict on test data
        y_pred = svr.predict(X_test)
        # Evaluate the model
        calculateMetrics("Support Vector Regressor", y_pred, y_test)
        return render(request,'prediction.html',
                    {'algorithm':'Support Vector',
                    'mse_list':mse_list[-1],
                    'r2_list':r2_list[-1],
                    'mae_list':mae_list[-1]})

import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

base_estimator=DecisionTreeRegressor()
def bagging_with_dt(request):
    # Bagging Regressor (with DecisionTreeRegressor as base estimator)
    if X_train is None:
        messages.error(request,'Please upload dataset first')
        return redirect('upload')
    else:
        if os.path.exists('static/Model/bagging_with_dt.pkl'):
            Bag_rf = joblib.load('static/Model/bagging_with_dt.pkl')
            print("Bagging Regressor model loaded successfully.")
        else:
            Bag_rf = BaggingRegressor(
                base_estimator=base_estimator,  # Passing DecisionTreeRegressor as the base estimator
                n_estimators=10,               # Number of trees (estimators)
                random_state=42,               # Ensures reproducibility
                                    # Use all available CPU cores
            )
            Bag_rf.fit(X_train, y_train)
            joblib.dump(Bag_rf, 'static/Model/bagging_with_dt.pkl')
            print("Bagging Regressor with Decision Trees saved successfully.")

        # Predictions and Evaluation
        y_pred_rf = Bag_rf.predict(X_test)
        calculateMetrics("Bagging Regressor (Decision Trees)", y_pred_rf, y_test)
        return render(request,'prediction.html',
                    {'algorithm':'Bagging Regressor (DT)',
                    'mae_list':mae_list[-1],
                    'mse_list':mse_list[-1],
                    'r2_list':r2_list[-1],
                    'rmse_list':rmse_list[-1]})

def prediction(request):
    Test=True
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        test = pd.read_csv(default_storage.path(file_path))
        Bag_rf = joblib.load('static/Model/bagging_with_dt.pkl')
        Final_output = Bag_rf.predict(test)
        Final_output
        pred=pd.DataFrame(Final_output) 
        test['predicted']=pred
        default_storage.delete(file_path)
        return render(request,'prediction.html',{'predict':test.to_html()}) 
    return render(request,'prediction.html',{'test':Test})