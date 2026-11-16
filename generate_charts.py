import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import LabelEncoder

os.makedirs('static/charts', exist_ok=True)
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

plt.figure(figsize=(6, 4))
rate_ot = df.groupby('OverTime')['Attrition'].apply(lambda s: (s == 'Yes').mean() * 100)
ax = rate_ot.plot(kind='bar', color=['#4C72B0', '#DD8452'], width=0.5)
ax.set_ylabel('Attrition Rate (%)', fontsize=11)
ax.set_xlabel('OverTime', fontsize=11)
ax.set_title('Attrition Rate by OverTime', fontsize=13, fontweight='bold', pad=10)
plt.xticks(rotation=0)
for i, v in enumerate(rate_ot):
    ax.text(i, v + 0.8, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=10)
plt.ylim(0, 35)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('static/charts/eda_overtime.png', dpi=150)
plt.close()

plt.figure(figsize=(6.5, 4))
rate_dept = df.groupby('Department')['Attrition'].apply(lambda s: (s == 'Yes').mean() * 100).sort_values()
ax = rate_dept.plot(kind='barh', color='#55A868', width=0.5)
ax.set_xlabel('Attrition Rate (%)', fontsize=11)
ax.set_ylabel('Department', fontsize=11)
ax.set_title('Attrition Rate by Department', fontsize=13, fontweight='bold', pad=10)
for i, v in enumerate(rate_dept):
    ax.text(v + 0.4, i, f'{v:.1f}%', va='center', fontweight='bold', fontsize=10)
plt.xlim(0, 25)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('static/charts/eda_department.png', dpi=150)
plt.close()

plt.figure(figsize=(6.5, 4.5))
sns.boxplot(data=df, x='Attrition', y='MonthlyIncome', hue='Attrition', palette=['#DD8452', '#4C72B0'], width=0.5, legend=False)
plt.title('Monthly Income: Stayed vs Left', fontsize=13, fontweight='bold', pad=10)
plt.xlabel('Attrition Status', fontsize=11)
plt.ylabel('Monthly Income ($)', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('static/charts/eda_income.png', dpi=150)
plt.close()

plt.figure(figsize=(5, 4.5))
counts = df['Attrition'].value_counts()
plt.pie(counts, labels=['Stayed (83.9%)', 'Left (16.1%)'], autopct='%1.1f%%', startangle=140, colors=['#55A868', '#DD8452'], explode=(0, 0.08), wedgeprops=dict(width=0.6, edgecolor='white'))
plt.title('Overall Employee Attrition Distribution', fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('static/charts/eda_attrition_pie.png', dpi=150)
plt.close()

numdf = df.select_dtypes(include=[np.number]).drop(columns=['EmployeeCount', 'StandardHours', 'EmployeeNumber'], errors='ignore')
plt.figure(figsize=(10, 8))
sns.heatmap(numdf.corr(), cmap='coolwarm', center=0, cbar_kws={'shrink': 0.7})
plt.title('Correlation Heatmap of Numeric Features', fontsize=13, fontweight='bold', pad=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig('static/charts/eda_heatmap.png', dpi=150)
plt.close()

df_prep = df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], errors='ignore')
df_prep['IncomePerYearAtCompany'] = df_prep['MonthlyIncome'] / (df_prep['YearsAtCompany'] + 1)
df_prep['TenureRatio'] = df_prep['YearsAtCompany'] / (df_prep['TotalWorkingYears'] + 1)
df_prep['PromotionGap'] = df_prep['YearsAtCompany'] - df_prep['YearsSinceLastPromotion']
df_prep['IsFrequentTraveler'] = (df_prep['BusinessTravel'] == 'Travel_Frequently').astype(int)

cat_cols = ['Attrition', 'BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
for col in cat_cols:
    le = LabelEncoder()
    df_prep[col] = le.fit_transform(df_prep[col])

X = df_prep.drop('Attrition', axis=1)
y = df_prep['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
}

rows = []
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    rows.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Macro F1': f1_score(y_test, pred, average='macro'),
        'Recall (Leavers)': recall_score(y_test, pred, pos_label=1)
    })
res_df = pd.DataFrame(rows).set_index('Model')

plt.figure(figsize=(7.5, 4.8))
ax = res_df.plot(kind='bar', figsize=(7.5, 4.8), color=['#4C72B0', '#DD8452', '#55A868'], width=0.7)
plt.ylim(0, 1.0)
plt.title('Model Comparison Across Key Metrics', fontsize=13, fontweight='bold', pad=10)
plt.ylabel('Score (0 - 1.0)', fontsize=11)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('static/charts/model_comparison.png', dpi=150)
plt.close()

rf = models['Random Forest']
pred_rf = rf.predict(X_test)
cm = confusion_matrix(y_test, pred_rf)
fig, ax = plt.subplots(figsize=(5, 4.2))
ConfusionMatrixDisplay(cm, display_labels=['Stayed', 'Left']).plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('Confusion Matrix - Random Forest', fontsize=12, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('static/charts/eval_confusion_matrix.png', dpi=150)
plt.close()

proba_rf = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, proba_rf)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(5.5, 4.2))
ax.plot(fpr, tpr, color='#4C72B0', lw=2.5, label=f'Random Forest (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1.2, linestyle='--')
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate', fontsize=10)
ax.set_title('ROC Curve - Random Forest', fontsize=12, fontweight='bold', pad=10)
ax.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('static/charts/eval_roc_curve.png', dpi=150)
plt.close()

importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(7.5, 4.8))
importance.head(10).sort_values().plot(kind='barh', color='#4C72B0', width=0.6)
plt.title('Top 10 Feature Importances (Random Forest)', fontsize=13, fontweight='bold', pad=10)
plt.xlabel('Importance Weight', fontsize=10)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('static/charts/eval_feature_importance.png', dpi=150)
plt.close()

print('All figures generated into static/charts/')
