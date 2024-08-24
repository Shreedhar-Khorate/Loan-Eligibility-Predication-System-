import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

file_path = 's.csv'
loan_data = pd.read_csv(file_path)

loan_data.dropna(inplace=True)

if loan_data['Loan_Status'].isna().any():
    print("NaN values found in 'Loan_Status'. Dropping rows with NaN in 'Loan_Status'.")
    loan_data.dropna(subset=['Loan_Status'], inplace=True)

loan_data['Loan_Status'] = loan_data['Loan_Status'].str.lower().map({'y': 1, 'n': 0})

if loan_data['Loan_Status'].isna().any():
    print("NaN values still present in 'Loan_Status' after mapping. Exiting.")
    exit()

label_encoders = {}
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

for column in categorical_columns:
    le = LabelEncoder()
    loan_data[column] = le.fit_transform(loan_data[column].str.lower()) 
    label_encoders[column] = le

X = loan_data.drop(columns=['Loan_ID', 'Loan_Status'])
y = loan_data['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if y_train.isna().any() or y_test.isna().any():
    print("NaN values found in training or testing target variable. Exiting.")
    exit()

scaler = StandardScaler()
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

def get_user_input(label_encoders, scaler):
    gender = input("Enter your gender (Male/Female): ").strip().lower()
    married = input("Are you married (Yes/No): ").strip().lower()
    dependents = input("Enter number of dependents (0, 1, 2, 3+): ").strip()
    education = input("Enter your education level (Graduate/Not Graduate): ").strip().lower()
    self_employed = input("Are you self-employed (Yes/No): ").strip().lower()
    property_area = input("Enter property area (Urban/Rural/Semiurban): ").strip().lower()

    income = float(input("Enter your income: "))
    coapplicant_income = float(input("Enter coapplicant's income: "))
    loan_amount = float(input("Enter the loan amount: "))
    loan_amount_term = float(input("Enter the loan amount term : "))
    credit_history = float(input("Enter your credit history: "))

    user_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'Property_Area': property_area,
        'ApplicantIncome': income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history
    }

    for column in categorical_columns:
        user_data[column] = [0]

    user_df = pd.DataFrame(user_data, index=[0])

    user_df = user_df[X.columns]

    user_df[numerical_columns] = scaler.transform(user_df[numerical_columns])

    return user_df

def predict_eligibility(model, user_data):
    prediction = model.predict(user_data)[0]
    if prediction == 1:
        return "You are eligible for the loan."
    else:
        return "You are not eligible for the loan."

user_data = get_user_input(label_encoders, scaler)
result = predict_eligibility(model, user_data)
print(result)