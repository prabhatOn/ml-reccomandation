from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from model import hybrid_recommendation
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["MONGO_URI"] = "mongodb+srv://prabhatchaubey56:helloworld@cluster0.cbhfk.mongodb.net/musicApp?retryWrites=true&w=majority"

mongo = PyMongo(app)
try:
    mongo.db.command('ping')  # Test the connection
    print("MongoDB Connection: Successful")
except Exception as e:
    print(f"MongoDB Connection Error: {str(e)}")

def create_collections():
    try:
        if mongo.db is not None:
            if 'users' not in mongo.db.list_collection_names():
                mongo.db.create_collection('users')
            if 'recommendations' not in mongo.db.list_collection_names():
                mongo.db.create_collection('recommendations')
            print("Collections ensured!")
    except Exception as e:
        print(f"Error while creating collections: {str(e)}")

# Call the function to create collections
create_collections()

@app.route('/')
def index():
    if 'email' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

# Signup Route
@app.route('/signup', methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        try:
            users = mongo.db.users
            existing_user = users.find_one({'email': request.form['email']})

            if existing_user is None:
                password_hash = generate_password_hash(request.form['password'])
                users.insert_one({
                    'name': request.form['name'],
                    'email': request.form['email'],
                    'password': password_hash
                })
                session['email'] = request.form['email']
                return redirect(url_for('dashboard'))
            else:
                return 'User already exists!'
        except Exception as e:
            return f"An error occurred during signup: {str(e)}"
    
    return render_template('signup.html')

# Login Route
@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        try:
            users = mongo.db.users
            login_user = users.find_one({'email': request.form['email']})

            if login_user and check_password_hash(login_user['password'], request.form['password']):
                session['email'] = request.form['email']
                return redirect(url_for('dashboard'))
            else:
                return 'Invalid email or password'
        except Exception as e:
            return f"An error occurred during login: {str(e)}"
    
    return render_template('login.html')

# Dashboard Route
@app.route('/dashboard')
def dashboard():
    if 'email' in session:
        return render_template('dashboard.html', email=session['email'])
    return redirect(url_for('index'))

# Recommendation Route
@app.route('/recommend', methods=['POST'])
def recommend():
    if 'email' not in session:
        return redirect(url_for('login'))
    user_id = request.form.get('user_id')  
    track_id = request.form.get('track_id') 
    mood = request.form.get('mood')  
    user_input = request.form.get('user_input')  
    recommendations = hybrid_recommendation(user_id=user_id, track_id=track_id, mood=mood, user_input=user_input, top_n=5)

    if recommendations is not None:
        mongo.db.recommendations.insert_one({
            'email': session['email'],
            'recommendations': recommendations.to_dict(orient='records'),
            'mood': mood  
        })
        return jsonify(recommendations.to_dict(orient='records'))
    else:
        return jsonify({'message': 'No recommendations available for the provided mood.'})

# Logout Route
@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
