import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def veri_hazirla(dosya_yolu):
    temp_df = pd.read_csv(dosya_yolu)
    temp_df.drop('customerID', axis=1, inplace=True)
    temp_df['TotalCharges'] = pd.to_numeric(temp_df['TotalCharges'], errors='coerce')
    temp_df.dropna(inplace=True)
    return temp_df


data_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
df_final = veri_hazirla(data_path)


df_final['Monthly_to_Total_Ratio'] = df_final['MonthlyCharges'] / df_final['TotalCharges']
df_final['Monthly_to_Total_Ratio'] = df_final['Monthly_to_Total_Ratio'].fillna(0)

print("Veri yüklendi. Satır/Sütun sayısı:", df_final.shape)
print("\nSayısal Değerlerin Özeti:\n", df_final.describe())

plt.figure(figsize=(10, 6))
sns.countplot(x='Contract', hue='Churn', data=df_final, palette='Set2')
plt.title('Sözleşme Türüne Göre Müşteri Terk Durumu')
plt.xlabel('Sözleşme Tipi')
plt.ylabel('Müşteri Sayısı')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='InternetService', hue='Churn', data=df_final, palette='viridis')
plt.title('İnternet Servis Türüne Göre Terk Durumu')
plt.show()


df_final['Churn'] = df_final['Churn'].map({'Yes': 1, 'No': 0})
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
binary_cols = [col for col in df_final.columns if df_final[col].dtype == 'object' and df_final[col].nunique() == 2]
for col in binary_cols:
    df_final[col] = le.fit_transform(df_final[col])
df_final = pd.get_dummies(df_final, drop_first=True)
print("\n 'Churn' sütunu var mı?", 'Churn' in df_final.columns)


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X = df_final.drop('Churn', axis=1)
y = df_final['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


dt_model = DecisionTreeClassifier(max_depth=7, class_weight='balanced', random_state=42)
dt_model.fit(X_train, y_train)
print("\n 1. Karar Ağacı Başarı Raporu")
print(classification_report(y_test, dt_model.predict(X_test)))

# 2. SVM
svm_model = LinearSVC(random_state=42, max_iter=2000, dual=False, class_weight='balanced')
svm_model.fit(X_train_scaled, y_train)
print("\n 2. SVM Başarı Raporu")
print(classification_report(y_test, svm_model.predict(X_test_scaled)))


rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight={0: 1, 1: 2}, random_state=42)
rf_model.fit(X_train, y_train)
print("\n 3. RANDOM FOREST Başarı Raporu")
print(classification_report(y_test, rf_model.predict(X_test)))



from sklearn.metrics import cohen_kappa_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score

print("\n" + "="*40)

dt_preds = dt_model.predict(X_test)
svm_preds = svm_model.predict(X_test_scaled)
rf_preds = rf_model.predict(X_test)


print(f"Karar Ağacı   -> Cohen's Kappa: {cohen_kappa_score(y_test, dt_preds):.4f} | ROC-AUC: {roc_auc_score(y_test, dt_preds):.4f}")
print(f"SVM           -> Cohen's Kappa: {cohen_kappa_score(y_test, svm_preds):.4f} | ROC-AUC: {roc_auc_score(y_test, svm_preds):.4f}")
print(f"Random Forest -> Cohen's Kappa: {cohen_kappa_score(y_test, rf_preds):.4f} | ROC-AUC: {roc_auc_score(y_test, rf_preds):.4f}")

print("\n CROSS VALIDATION")
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print(f"Her bir katmanın skoru: {np.round(cv_scores, 4)}")
print(f"Ortalama Güvenilirlik Skoru: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")


plt.figure(figsize=(10, 6))
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh', color='darkred', edgecolor='black')
plt.title("Feature Importance - Random Forest")
plt.xlabel("Etki Oranı (Ağırlık)")
plt.ylabel("Değişkenler")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Kalacak (0)', 'Gidecek (1)'],
            yticklabels=['Kalacak (0)', 'Gidecek (1)'])
plt.title("Random Forest - Karmaşıklık Matrisi (Confusion Matrix")
plt.ylabel('Gerçek Durum')
plt.xlabel('Modelin Tahmini')
plt.tight_layout()
plt.show()