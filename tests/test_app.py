import io
import json
import numpy as np
import pytest

from app import create_app, db
from app.models import User, Prediction


@pytest.fixture
def app():
    app = create_app()
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    with app.app_context():
        db.create_all()
        from app.models import create_default_admin
        create_default_admin()
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def login(client, username, password):
    return client.post('/login', data={'username': username, 'password': password}, follow_redirects=True)


def create_user(client, username='testuser', password='pass123', first='John', last='Doe', role='user'):
    return client.post('/admin/create-user', data={
        'username': username, 'password': password,
        'first_name': first, 'last_name': last, 'role': role,
    }, follow_redirects=True)


# ── Auth ──────────────────────────────────────────────────────────────────────

def test_login_page_loads(client):
    r = client.get('/login')
    assert r.status_code == 200


def test_login_success(client):
    r = login(client, 'admin', 'admin123')
    assert r.status_code == 200


def test_login_wrong_password(client):
    r = login(client, 'admin', 'wrong')
    assert b'Invalid' in r.data


def test_redirect_unauthenticated(client):
    r = client.get('/profile')
    assert r.status_code == 302


def test_logout(client):
    login(client, 'admin', 'admin123')
    r = client.get('/logout', follow_redirects=True)
    assert r.status_code == 200


# ── Admin ─────────────────────────────────────────────────────────────────────

def test_admin_create_user(client):
    login(client, 'admin', 'admin123')
    r = create_user(client)
    assert r.status_code == 200
    with client.application.app_context():
        assert User.query.filter_by(username='testuser').first() is not None


def test_admin_create_duplicate_user(client):
    login(client, 'admin', 'admin123')
    create_user(client, username='dup')
    r = create_user(client, username='dup')
    assert r.status_code == 200


def test_non_admin_cannot_create_user(client, app):
    with app.app_context():
        u = User(username='normaluser', first_name='A', last_name='B', role='user')
        u.set_password('pass')
        db.session.add(u)
        db.session.commit()
    login(client, 'normaluser', 'pass')
    r = client.post('/admin/create-user', data={
        'username': 'x', 'password': 'x', 'first_name': 'x', 'last_name': 'x'
    })
    assert r.status_code == 403


def test_admin_page_requires_admin_role(client, app):
    with app.app_context():
        u = User(username='normaluser2', first_name='A', last_name='B', role='user')
        u.set_password('pass')
        db.session.add(u)
        db.session.commit()
    login(client, 'normaluser2', 'pass')
    r = client.get('/admin', follow_redirects=True)
    assert b'Access denied' in r.data


# ── Label repair ──────────────────────────────────────────────────────────────

def test_label_repair():
    raw = [
        'a' * 32 + 'Gliese_667',
        'b' * 32 + 'Kepler_442',
        'c' * 32 + 'Gliese_667',
        'd' * 32 + 'Proxima_b',
    ]
    label_map = {name: i for i, name in enumerate(sorted(set(s[32:] for s in raw)))}
    assert label_map['Gliese_667'] == 0
    assert label_map['Kepler_442'] == 1
    assert label_map['Proxima_b'] == 2
    assert len(label_map) == 3


def test_label_repair_consistent():
    raw = ['x' * 32 + 'Alpha'] * 5 + ['y' * 32 + 'Beta'] * 5
    label_map = {name: i for i, name in enumerate(sorted(set(s[32:] for s in raw)))}
    assert label_map['Alpha'] == 0
    assert label_map['Beta'] == 1


# ── Model ─────────────────────────────────────────────────────────────────────

def test_model_loads():
    from app.ml import load_model
    model, norm, label_map, label_map_inv = load_model()
    assert model is not None
    assert len(label_map) > 0


def test_label_map_inv_consistent():
    from app.ml import load_model
    _, _, label_map, label_map_inv = load_model()
    for name, idx in label_map.items():
        assert label_map_inv[idx] == name


# ── API endpoints ─────────────────────────────────────────────────────────────

def test_api_training_log(client):
    login(client, 'admin', 'admin123')
    r = client.get('/api/training-log')
    assert r.status_code == 200
    assert isinstance(json.loads(r.data), list)


def test_api_class_distribution(client):
    login(client, 'admin', 'admin123')
    r = client.get('/api/class-distribution')
    assert r.status_code == 200
    assert isinstance(json.loads(r.data), dict)


def test_api_validation_top5(client):
    login(client, 'admin', 'admin123')
    r = client.get('/api/validation-top5')
    assert r.status_code == 200
    assert len(json.loads(r.data)) <= 5


def test_api_last_prediction_no_data(client):
    login(client, 'admin', 'admin123')
    r = client.get('/api/last-prediction')
    assert r.status_code == 404


def test_api_requires_auth(client):
    for ep in ['/api/training-log', '/api/class-distribution', '/api/validation-top5', '/api/last-prediction']:
        assert client.get(ep).status_code == 302


# ── Upload ────────────────────────────────────────────────────────────────────

def test_upload_no_file(client):
    login(client, 'admin', 'admin123')
    r = client.post('/upload', data={}, follow_redirects=True)
    assert b'No file' in r.data


def test_upload_npy(client):
    login(client, 'admin', 'admin123')
    buf = io.BytesIO()
    np.save(buf, np.random.rand(2, 80000).astype(np.float32))
    buf.seek(0)
    r = client.post('/upload', data={'file': (buf, 'test.npy')},
                    content_type='multipart/form-data', follow_redirects=True)
    assert r.status_code == 200


def test_upload_stores_prediction(client, app):
    login(client, 'admin', 'admin123')
    buf = io.BytesIO()
    np.save(buf, np.random.rand(2, 80000).astype(np.float32))
    buf.seek(0)
    client.post('/upload', data={'file': (buf, 'test.npy')},
                content_type='multipart/form-data', follow_redirects=True)
    with app.app_context():
        admin = User.query.filter_by(username='admin').first()
        assert len(Prediction.query.filter_by(user_id=admin.id).all()) >= 1
