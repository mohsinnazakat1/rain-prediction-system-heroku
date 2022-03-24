from flask import Flask,flash, render_template, request, redirect, url_for
import pickle5 as pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd


app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

@app.route("/")
def home():
	return render_template("index.html")

@app.route("/", methods = ['POST'])
def predict():
	Location = request.form.get("Location")
	WindGustDir = request.form.get("WindGustDir")
	WindDir9am = request.form.get("WindDir9am")
	WindDir3pm = request.form.get("WindDir3pm")

	if Location and WindGustDir and WindDir3pm and WindDir9am is not '' :
		Location_pkl_file = open('./models/Location_encoder.pkl', 'rb')
		Location_encoder = pickle.load(Location_pkl_file) 
		Location_pkl_file.close()
		Location_label = Location_encoder.transform([Location])

		WindGustDir_pkl_file = open('./models/WindGustDir_encoder.pkl', 'rb')
		WindGustDir_encoder = pickle.load(WindGustDir_pkl_file) 
		WindGustDir_pkl_file.close()
		WindGustDir_label = WindGustDir_encoder.transform([WindGustDir])

		WindDir9am_pkl_file = open('./models/WindDir9am_encoder.pkl', 'rb')
		WindDir9am_encoder = pickle.load(WindDir9am_pkl_file) 
		WindDir9am_pkl_file.close()
		WindDir9am_label = WindDir9am_encoder.transform([WindDir9am])

		WindDir3pm_pkl_file = open('./models/WindDir3pm_encoder.pkl', 'rb')
		WindDir3pm_encoder = pickle.load(WindDir3pm_pkl_file) 
		WindDir3pm_pkl_file.close()
		WindDir3pm_label = WindDir3pm_encoder.transform([WindDir3pm])

		user_input = pd.DataFrame({ 'Location': Location_label,'WindGustDir': WindGustDir_label,'WindDir9am': WindDir9am_label,'WindDir3pm': WindDir3pm_label})

		model = pickle.load(open('./models/svc_trained_model_03.pkl', 'rb'))

		predicted_rain = model.predict(user_input)
		prediction = ""
		if(predicted_rain == 1): 
			prediction = "Rain"
		if(predicted_rain == 0):
			prediction = "No Rain"

		return redirect(url_for('home', prediction=prediction))
	else:
		flash('Please enter all feilds.','negative')
		return redirect(url_for('home'))

if __name__ == "__main__":
  app.run()