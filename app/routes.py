from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from app import db
from app.models import User, Prediction, TrainingLog
from app import ml
import numpy as np
import json

bp = Blueprint('routes', __name__)

@bp.route('/', methods=['GET'])
def index():
    if current_user.is_authenticated:
        return redirect(url_for('routes.profile'))
    return redirect(url_for('routes.login'))

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('routes.profile'))
        flash('Invalid username or password')
    return render_template('login.html')

@bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('routes.login'))

@bp.route('/profile')
@login_required
def profile():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    return render_template('profile.html', predictions=predictions)

@bp.route('/admin')
@login_required
def admin():
    if current_user.role != 'admin':
        flash('Access denied')
        return redirect(url_for('routes.profile'))
    users = User.query.all()
    return render_template('admin.html', users=users)

@bp.route('/admin/create-user', methods=['POST'])
@login_required
def create_user():
    if current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    username = request.form.get('username')
    password = request.form.get('password')
    first_name = request.form.get('first_name')
    last_name = request.form.get('last_name')
    role = request.form.get('role', 'user')
    
    if User.query.filter_by(username=username).first():
        flash('Username already exists')
        return redirect(url_for('routes.admin'))
    
    user = User(username=username, first_name=first_name, last_name=last_name, role=role)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    
    flash('User created successfully')
    return redirect(url_for('routes.admin'))

@bp.route('/admin/delete-user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    user = User.query.get(user_id)
    if user and user.username != 'admin':
        db.session.delete(user)
        db.session.commit()
        flash('User deleted')
    return redirect(url_for('routes.admin'))

@bp.route('/analytics')
@login_required
def analytics():
    training_log = ml.get_training_log()
    return render_template('analytics.html', training_log=training_log)

@bp.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('routes.profile'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('routes.profile'))
    
    try:
        # Allow pickle for object arrays; handle both .npy and .npz
        raw = np.load(file, allow_pickle=True)
        # .npz → dict-like; .npy → array
        if hasattr(raw, 'files'):
            # NPZ file — try common key names
            keys = raw.files
            x_key = next((k for k in ['test_x', 'valid_x', 'train_x'] if k in keys), None)
            y_key = next((k for k in ['test_y', 'valid_y', 'train_y'] if k in keys), None)
            if x_key is None:
                flash('NPZ file must contain test_x, valid_x, or train_x array')
                return redirect(url_for('routes.profile'))
            audio_data = raw[x_key]
            labels = raw[y_key] if y_key else None
        else:
            audio_data = raw
            labels = None
        
        results = ml.predict_batch(audio_data)
        
        accuracy = None
        loss = None
        if labels is not None:
            with open('model/label_map.json', 'r') as f:
                lm = json.load(f)
            true_ids = []
            for l in labels:
                l_str = str(l)
                if len(l_str) > 32 and l_str[:32].isalnum():
                    name = l_str[32:]
                    true_ids.append(lm.get(name, -1))
                else:
                    try:
                        true_ids.append(int(l_str))
                    except ValueError:
                        true_ids.append(lm.get(l_str, -1))
            correct = sum(1 for r, t in zip(results, true_ids) if r['class_id'] == t)
            accuracy = correct / len(true_ids)
        
        prediction = Prediction(
            user_id=current_user.id,
            filename=file.filename,
            accuracy=accuracy,
            loss=loss,
            results=json.dumps(results)
        )
        db.session.add(prediction)
        db.session.commit()
        
        flash(f'File processed. Accuracy: {accuracy:.2%}' if accuracy else 'File processed')
    except Exception as e:
        flash(f'Error processing file: {str(e)}')
    
    return redirect(url_for('routes.profile'))

@bp.route('/api/training-log')
@login_required
def api_training_log():
    log = ml.get_training_log()
    return jsonify(log)

@bp.route('/api/class-distribution')
@login_required
def api_class_distribution():
    distribution = ml.get_class_distribution()
    return jsonify(distribution)

@bp.route('/api/validation-top5')
@login_required
def api_validation_top5():
    with open('model/label_map.json', 'r') as f:
        label_map = json.load(f)
    n_per = 400 // len(label_map)
    counts = {name: n_per for name in label_map.keys()}
    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5])
    return jsonify(sorted_counts)

@bp.route('/api/last-prediction')
@login_required
def api_last_prediction():
    prediction = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).first()
    if not prediction or not prediction.results:
        return jsonify({'error': 'No prediction results available'}), 404
    results = json.loads(prediction.results)
    return jsonify({
        'filename': prediction.filename,
        'accuracy': prediction.accuracy,
        'loss': prediction.loss,
        'results': results,
        'timestamp': prediction.timestamp.isoformat()
    })
