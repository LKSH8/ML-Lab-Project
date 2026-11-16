from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    saved = pickle.load(f)

model = saved['model']
encoders = saved['encoders']
FEATURE_COLUMNS = saved['feature_columns']
FEATURE_IMPORTANCE = saved.get('feature_importance', {})

df_raw = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

DATASET_STATS = {
    'total_records': f"{len(df_raw):,}",
    'num_features': df_raw.shape[1],
    'attrition_yes': int((df_raw['Attrition'] == 'Yes').sum()),
    'attrition_no': int((df_raw['Attrition'] == 'No').sum()),
    'attrition_rate': f"{(df_raw['Attrition'] == 'Yes').mean() * 100:.1f}%",
    'avg_age': f"{df_raw['Age'].mean():.1f} years",
    'avg_income': f"${df_raw['MonthlyIncome'].mean():,.0f}",
    'avg_tenure': f"{df_raw['YearsAtCompany'].mean():.1f} years",
    'overtime_pct': f"{(df_raw['OverTime'] == 'Yes').mean() * 100:.1f}%",
}

PREVIEW_COLUMNS = [
    'Age', 'Attrition', 'BusinessTravel', 'Department', 'JobRole',
    'MonthlyIncome', 'OverTime', 'TotalWorkingYears', 'YearsAtCompany', 'JobSatisfaction'
]
PREVIEW_DATA = df_raw[PREVIEW_COLUMNS].head(12).to_dict(orient='records')

VISIBLE_CATEGORICAL = ['BusinessTravel', 'OverTime']
VISIBLE_NUMERIC = ['MonthlyIncome', 'Age', 'TotalWorkingYears',
                    'YearsAtCompany', 'YearsSinceLastPromotion']

VISIBLE_CATEGORICAL_OPTIONS = {col: list(encoders[col].classes_) for col in VISIBLE_CATEGORICAL}

FIELD_HINTS = {
    'MonthlyIncome': 'monthly salary ($) — daily and hourly rates are computed automatically',
    'Age': 'years',
    'TotalWorkingYears': 'years, entire professional career',
    'YearsAtCompany': 'years spent at this company',
    'YearsSinceLastPromotion': 'years since last promotion',
}

DEFAULT_NUMERIC = {
    'DistanceFromHome': 7, 'Education': 3, 'EnvironmentSatisfaction': 3, 'JobInvolvement': 3,
    'JobLevel': 2, 'JobSatisfaction': 3, 'MonthlyRate': 14235, 'NumCompaniesWorked': 2,
    'PercentSalaryHike': 14, 'PerformanceRating': 3, 'RelationshipSatisfaction': 3,
    'StockOptionLevel': 1, 'TrainingTimesLastYear': 3, 'WorkLifeBalance': 3,
    'YearsInCurrentRole': 3, 'YearsWithCurrManager': 3,
}
DEFAULT_CATEGORICAL = {
    'Department': 'Research & Development', 'EducationField': 'Life Sciences',
    'Gender': 'Male', 'JobRole': 'Sales Executive', 'MaritalStatus': 'Married',
}

FRIENDLY_NAMES = {
    'MonthlyIncome': 'Monthly Income',
    'Age': 'Age',
    'OverTime': 'Overtime',
    'TotalWorkingYears': 'Total Working Years',
    'DailyRate': 'Daily Rate',
    'IncomePerYearAtCompany': 'Income per Year at Company',
    'TenureRatio': 'Tenure Ratio',
    'YearsAtCompany': 'Years at Company',
    'PromotionGap': 'Promotion Gap',
    'HourlyRate': 'Hourly Rate',
}

def build_feature_row(form):
    row = {}
    for col in VISIBLE_CATEGORICAL:
        row[col] = encoders[col].transform([form[col]])[0]
    for col in VISIBLE_NUMERIC:
        row[col] = float(form[col])

    monthly_inc = row['MonthlyIncome']
    daily_calc = float(form.get('DailyRate') or round(monthly_inc / 21.75, 2))
    hourly_calc = float(form.get('HourlyRate') or round(daily_calc / 8.0, 2))
    row['DailyRate'] = daily_calc
    row['HourlyRate'] = hourly_calc

    for col, val in DEFAULT_NUMERIC.items():
        if col not in row:
            row[col] = val
    for col, val in DEFAULT_CATEGORICAL.items():
        if col not in row:
            row[col] = encoders[col].transform([val])[0]

    row['IncomePerYearAtCompany'] = row['MonthlyIncome'] / (row['YearsAtCompany'] + 1)
    row['TenureRatio'] = row['YearsAtCompany'] / (row['TotalWorkingYears'] + 1)
    row['PromotionGap'] = row['YearsAtCompany'] - row['YearsSinceLastPromotion']
    row['IsFrequentTraveler'] = 1 if form['BusinessTravel'] == 'Travel_Frequently' else 0

    return [row[col] for col in FEATURE_COLUMNS]

@app.route('/', methods=['GET'])
def home():
    return render_template(
        'index.html',
        active_tab='predict',
        numeric_fields=VISIBLE_NUMERIC,
        categorical_options=VISIBLE_CATEGORICAL_OPTIONS,
        hints=FIELD_HINTS,
        result=None,
        stats=DATASET_STATS,
        preview_columns=PREVIEW_COLUMNS,
        preview_data=PREVIEW_DATA,
    )

@app.route('/dataset', methods=['GET'])
def dataset():
    return render_template(
        'index.html',
        active_tab='dataset',
        numeric_fields=VISIBLE_NUMERIC,
        categorical_options=VISIBLE_CATEGORICAL_OPTIONS,
        hints=FIELD_HINTS,
        result=None,
        stats=DATASET_STATS,
        preview_columns=PREVIEW_COLUMNS,
        preview_data=PREVIEW_DATA,
    )

@app.route('/models', methods=['GET'])
def models():
    return render_template(
        'index.html',
        active_tab='models',
        numeric_fields=VISIBLE_NUMERIC,
        categorical_options=VISIBLE_CATEGORICAL_OPTIONS,
        hints=FIELD_HINTS,
        result=None,
        stats=DATASET_STATS,
        preview_columns=PREVIEW_COLUMNS,
        preview_data=PREVIEW_DATA,
    )

@app.route('/predict', methods=['POST'])
def predict():
    row = build_feature_row(request.form)
    proba = model.predict_proba(np.array(row).reshape(1, -1))[0]
    prediction = int(np.argmax(proba))
    confidence = round(proba[prediction] * 100, 1)

    result = {
        'label': 'Likely to Leave' if prediction == 1 else 'Likely to Stay',
        'confidence': confidence,
        'is_risk': prediction == 1,
        'top_factors': [FRIENDLY_NAMES.get(k, k) for k in FEATURE_IMPORTANCE.keys()][:5],
    }

    return render_template(
        'index.html',
        active_tab='predict',
        numeric_fields=VISIBLE_NUMERIC,
        categorical_options=VISIBLE_CATEGORICAL_OPTIONS,
        hints=FIELD_HINTS,
        result=result,
        stats=DATASET_STATS,
        preview_columns=PREVIEW_COLUMNS,
        preview_data=PREVIEW_DATA,
    )

if __name__ == '__main__':
    app.run(debug=True)
