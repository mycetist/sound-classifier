# sound-classifier

Нейронная сеть для классификации звуковых сигналов

## Запуск

1. Положите файл `Data.npz` в корень проекта
2. Запустите следующие команды:

```bash
pip install -r requirements.txt
python train_v2.py # обучение модели
python seed_data.py # создает тестовые аккаунты
python run.py
```

Сервис будет поднят на http://localhost:5001

## Тестовые данные

| Username | Password | Роль |
|----------|----------|------|
| admin    | admin123 | admin |
| user     | user123  | user |

## Структура

```
app/           # Flask бэкэнд
templates/     # Jinja2 + Tailwind фронтэнд
model/         # данные о модели
notebooks/     # обучение модели
```
