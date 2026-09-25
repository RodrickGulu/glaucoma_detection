import os
import sqlite3
from flask import g, current_app
from werkzeug.security import check_password_hash, generate_password_hash

# Connect to SQLite database
DATABASE = os.environ.get('DATABASE_PATH', 'users.db')

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

def close_db():
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()
        g.pop('_database', None)

def init_db():
    db = get_db()
    with current_app.open_resource('schema.sql', mode='r') as f:
        db.cursor().executescript(f.read())
    db.commit()

def is_database_empty():
    db = get_db()
    cur = db.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM users")
        count = cur.fetchone()[0]
    except sqlite3.OperationalError as e:
        if 'no such table' in str(e).lower():
            init_db()
            return True
        raise
    return count == 0

# Mock user database for demonstration purposes
def authenticate(username, password):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cur.fetchone()
    if not user:
        return False
    return check_password_hash(user[3], password)

def add_user(full_name, username, password):
    db = get_db()
    cur = db.cursor()
    try:
        password_hash = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
        cur.execute("INSERT INTO users (full_names, username, password) VALUES (?, ?, ?)", (full_name, username, password_hash))
        db.commit()
        print("User added successfully:", username)
    except Exception as e:
        print("Error adding user:", e)
