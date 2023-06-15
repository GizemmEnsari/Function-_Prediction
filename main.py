import os.path
import matplotlib
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, mean_absolute_error

from tensorflow.keras.callbacks import EarlyStopping

matplotlib.use('Agg')

app = Flask(__name__)

@app.route('/image')
def show_img():
    image_path = os.path.join('static','img', 'prediction.png')
    return render_template('image.html',image_path = image_path)





@app.route('/', methods=['GET','POST'])
def index():
    if request.method =='POST':
        file = request.files['file']
        #read/check the existance of the uploaded file
        if file:
            # check if it is a tsv or a csv file
            if file.filename.endswith('.tsv'):
                data = pd.read_csv(file,sep='\t',header=None, names=['1','2'])

            elif file.filename.endswith('.csv'):
                data = pd.read_csv(file, header=None, names=['1', '2'])

            else:
                return render_template('index.html', error='Please upload a tsv or csv file')


            # split the data into training and testing sets
            x = data['1'].astype(float).values.reshape(-1,1)
            y = data['2'].astype(float).values.reshape(-1,1)
            x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

            # normalize the data
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            # creating the model
            model = Sequential()
            model.add(Dense(32,activation='relu', input_shape=(1,) ))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            # defining cross validation
            kFold = KFold(n_splits=5, shuffle=True, random_state= 42)

            # initialize lists for evaluation metrics
            loss_scores = []
            accuracy_scores = []
            r2_scores = []
            mae_scores = []

            # loop over folds
            for train_idx, test_idx in kFold.split(x):
                # get training and testing sets for this fold
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # train the model on this fold's training set
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2,
                                    callbacks=[early_stopping])

                # evaluate the model on this fold's testing set
                result = model.evaluate(x_test, y_test)
                loss = result if isinstance(result, float) else result[0]
                accuracy = result if isinstance(result, float) else result[1]
                loss_scores.append(loss)
                accuracy_scores.append(accuracy)







                # predict the output for this fold's testing set
                y_pred = model.predict(x_test)

                # calculate the R-squared and MAE for this fold's testing set
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2_scores.append(r2)
                mae_scores.append(mae)

            # calculate the average evaluation metrics over all folds
            avg_loss = np.mean(loss_scores)
            avg_accuracy = np.mean(accuracy_scores)
            avg_r2 = np.mean(r2_scores)
            avg_mae = np.mean(mae_scores)

            # print the evaluation results
            print("Average test set loss:", avg_loss)
            print("Average test set accuracy:", avg_accuracy)
            print("Average test set R-squared:", avg_r2)
            print("Average test set MAE:", avg_mae)


            early_stopping = EarlyStopping(monitor = 'val_loss', patience=10,restore_best_weights=True)
            history = model.fit(x_train,y_train,epochs=100,batch_size=32,validation_split=0.2,
                                callbacks=[early_stopping])

            loss = model.evaluate(x_test,y_test)

            x_pred = np.linspace(min(x),max(x),num=100).reshape(-1,1)
            y_pred = model.predict(scaler.transform((x_pred)))
            plt.scatter(x,y)
            plt.plot(x_pred,y_pred,color='red')
            plt.title('Function Prediction Results')
            plt.ylabel('y')
            plt.xlabel('x')
            plt.savefig('static/img/prediction.png')
            plt.close()

            result = model.evaluate(x_test, y_test)
            loss = result if isinstance(result, float) else result[0]
            accuracy = result if isinstance(result, float) else result[1]

            # Predict the output for the test set
            y_pred = model.predict(x_test)

            # Calculate the R-squared and MAE for the test set
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # Print the evaluation results
            print("Test set loss:", loss)
            print("Test set accuracy:", accuracy)
            print("Test set R-squared:", r2)
            print("Test set MAE:", mae)
            return redirect(url_for('show_img'))
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)