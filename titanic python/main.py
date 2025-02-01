import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_curve, auc,
                             precision_recall_curve, classification_report)

# Загрузка данных
data = pd.read_csv('titanic.csv')

# Проверяем названия столбцов
print("Колонки в датасете:", data.columns)

# Предобработка данных
if 'Age' in data.columns:
    data['Age'] = data['Age'].fillna(data['Age'].mean())
if 'Sex' in data.columns:
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Проверяем, есть ли нужные колонки
required_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    raise KeyError(f"В датасете отсутствуют колонки: {missing_columns}")

# Определяем признаки и целевую переменную
X = data[required_columns]
y = data['Survived']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказания
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Визуализация важности признаков
plt.figure(figsize=(10, 6))
plt.bar(X.columns, model.feature_importances_)
plt.title("Feature Importances")
plt.ylabel("Importance")
plt.xlabel("Feature")
plt.xticks(rotation=45)
plt.show()

# Кривая обучения
train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label="Training Score", color="blue")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Cross-validation Score", color="green")
plt.title("Learning Curve")
plt.xlabel("Number of Training Samples")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Did not Survive', 'Survived'], yticklabels=['Did not Survive', 'Survived'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC-кривая
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Precision-Recall кривая
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()
plt.show()

# Прогноз для нескольких новых пассажиров
new_data = pd.DataFrame({
    'Pclass': [1, 3, 2],
    'Sex': [0, 1, 0],  # Мужчина, Женщина, Мужчина
    'Age': [25, 30, 40],
    'SibSp': [0, 1, 0],
    'Parch': [0, 2, 1],
    'Fare': [100, 20, 50]
})

predictions = model.predict(new_data)
for i, pred in enumerate(predictions):
    print(f"Passenger {i+1}: {'Survived' if pred == 1 else 'Did not survive'}")
