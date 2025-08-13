# 📊 Предсказание оттока клиентов телеком-компании | Customer Churn Prediction for a Telecom Company

## 🔗 Project Notebook
[Google Colab Link](https://colab.research.google.com/drive/154vd8yHvzEQjEw0fmMCnOWGB1aodK--C)

## 🇷🇺 О проекте

### 🎯 Цель
Построить модель, которая определяет, уйдёт ли клиент, на основе демографических данных, типов контрактов и поведения при оплате.

### 💡 Почему это важно
Отток клиентов — одна из ключевых проблем в телеком-бизнесе. Удержание одного клиента обходится дешевле, чем привлечение нового.  
Модель помогает выявлять клиентов с высокой вероятностью ухода, чтобы компания могла вовремя предложить персонализированные акции и снизить потери.

### 📌 Что было сделано
- Проведён **EDA** — анализ причин оттока, географии и поведения клиентов.
- Обработаны пропуски и преобразованы категориальные признаки.
- Учтён дисбаланс классов с помощью `class_weights`.
- Обучена модель **CatBoostClassifier** (без кодирования категорий).
- Построена **ROC-кривая** и рассчитаны метрики.

### 🛠 Используемые технологии
- Python, pandas, NumPy  
- CatBoost  
- Scikit-learn  
- Matplotlib, Seaborn  

### 📊 Результаты модели
| Метрика    | Значение  |
|------------|-----------|
| Recall     | 80.75%    |
| Accuracy   | 74.95%    |
| F1 Score   | 63.11%    |
| ROC AUC    | 85.03%    |

Модель определяет ~81% клиентов, которые действительно могут уйти.

### 📂 Данные
- Источник: [Kaggle — Telco Customer Churn Dataset](https://www.kaggle.com/datasets/abdallahwagih/telco-customer-churn)  
- 7043 строк, 33 признака (24 категориальных)  
- Содержит: причины ухода, CLTV, геоданные, тип оплаты и контракта, историю использования услуг.

### 📈 Выводы
Модель может использоваться бизнесом для:
- автоматической оценки риска ухода;
- запуска персонализированных удерживающих кампаний;
- снижения общего уровня оттока.

---

## 🇬🇧 About the Project

### 🎯 Goal
Build a model to predict whether a customer will churn based on demographic data, contract types, and payment behavior.

### 💡 Why It Matters
Customer churn is one of the biggest challenges in the telecom industry.  
Retaining a customer is cheaper than acquiring a new one.  
This model identifies customers at high risk of leaving, enabling proactive retention actions.

### 📌 What Was Done
- Performed **EDA** — analysis of churn reasons, geography, and customer behavior.
- Handled missing values and transformed categorical features.
- Addressed class imbalance using `class_weights`.
- Trained a **CatBoostClassifier** (without manual categorical encoding).
- Built an **ROC curve** and calculated metrics.

### 🛠 Technologies Used
- Python, pandas, NumPy  
- CatBoost  
- Scikit-learn  
- Matplotlib, Seaborn  

### 📊 Model Results
| Metric     | Value    |
|------------|----------|
| Recall     | 80.75%   |
| Accuracy   | 74.95%   |
| F1 Score   | 63.11%   |
| ROC AUC    | 85.03%   |

The model identifies ~81% of customers who are likely to leave.

### 📂 Dataset
- Source: [Kaggle — Telco Customer Churn Dataset](https://www.kaggle.com/datasets/abdallahwagih/telco-customer-churn)  
- 7043 rows, 33 features (24 categorical)  
- Includes churn reasons, CLTV, geolocation, payment and contract type, service usage history.

### 📈 Conclusions
The model can be used for:
- Automatic churn risk assessment;
- Launching targeted retention campaigns;
- Reducing overall churn rate.

---



