from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from services.recommendation_service import hybrid_recommendation
from utils.dataLoader import load_dataset
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["MONGO_URI"] = os.getenv("MONGO_URI") 

if not app.config["MONGO_URI"]:
    raise ValueError("MONGO_URI environment variable not set.")

mongo = PyMongo(app)

# Load dataset once when the application starts
df = load_dataset()

def create_collections():
    if mongo.db is not None:
        if 'users' not in mongo.db.list_collection_names():
            mongo.db.create_collection('users')
        if 'recommendations' not in mongo.db.list_collection_names():
            mongo.db.create_collection('recommendations')

create_collections()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'email': request.form['email']})

        if existing_user is None:
            try:
                password_hash = generate_password_hash(request.form['password'])
                users.insert_one({'name': request.form['name'], 'email': request.form['email'], 'password': password_hash})
                session['email'] = request.form['email']
                return redirect(url_for('dashboard'))
            except Exception as e:
                return f"An error occurred during signup: {str(e)}"
        else:
            return 'User already exists!'
    return render_template('signup.html')

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        users = mongo.db.users
        login_user = users.find_one({'email': request.form['email']})

        if login_user and check_password_hash(login_user['password'], request.form['password']):
            session['email'] = request.form['email']
            return redirect(url_for('dashboard'))
        else:
            return 'Invalid email or password'
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'email' in session:
        return render_template('dashboard.html', email=session['email'])
    return redirect(url_for('index'))

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'email' not in session:
        return redirect(url_for('login'))

    track_id = request.form.get('track_id')
    mood = request.form.get('mood')
    user_input = request.form.get('user_input')
    recommendations = hybrid_recommendation(df, track_id=track_id, mood=mood, user_input=user_input, top_n=5)
    
    if recommendations is not None:
        mongo.db.recommendations.insert_one({
            'email': session['email'],
            'recommendations': recommendations.to_dict(orient='records'),
            'mood': mood  
        })
        return render_template('dashboard.html', email=session['email'], user_songs=recommendations.to_dict(orient='records'))
    else:
        return render_template('dashboard.html', email=session['email'], user_songs=None, message='No recommendations available for the provided mood.')

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
