from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# -------------------------------
# 🔹 BETTER DATASET (MORE REAL)
# -------------------------------

data = {
    'Income': [20000,30000,40000,50000,60000,70000,80000,90000,100000,120000],
    'Loan_Amount': [5000,10000,15000,20000,25000,30000,20000,15000,40000,50000],
    'Credit_Score': [500,600,650,700,720,750,780,800,820,850],
    'Employment': ['No','No','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes'],
    'Loan_Status': ['Rejected','Rejected','Approved','Approved','Approved','Approved','Approved','Approved','Approved','Approved']
}

df = pd.DataFrame(data)

# 🔹 Feature Engineering (IMPORTANT)
df['Loan_Income_Ratio'] = df['Loan_Amount'] / df['Income']

# Encode
le = LabelEncoder()
df['Employment'] = le.fit_transform(df['Employment'])
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

# Features
X = df[['Income', 'Loan_Amount', 'Credit_Score', 'Employment', 'Loan_Income_Ratio']]
y = df['Loan_Status']

# 🔹 Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 🔹 Model (stable)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# -------------------------------
# 🔹 HTML UI
# -------------------------------

html = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Loan Approval</title>
    <style>
        body {
            background: url('https://images.unsplash.com/photo-1560518883-ce09059eeffa');
            background-size: cover;
            font-family: Arial;
            text-align: center;
            color: white;
        }
        .box {
            margin-top: 100px;
            background: rgba(0,0,0,0.7);
            padding: 30px;
            display: inline-block;
            border-radius: 15px;
        }
        input {
            padding: 10px;
            margin: 10px;
            width: 220px;
            border-radius: 5px;
            border: none;
        }
        button {
            padding: 10px 20px;
            background: green;
            color: white;
            border: none;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="box">
        <h2>💰 AI Loan Approval System</h2>
        <form method="POST">
            <input type="number" name="income" placeholder="Income" required><br>
            <input type="number" name="loan" placeholder="Loan Amount" required><br>
            <input type="number" name="credit" placeholder="Credit Score" required><br>
            <input type="number" name="employment" placeholder="Employment (1=Yes, 0=No)" required><br>
            <button type="submit">Predict</button>
        </form>
        <h2>{{result}}</h2>
    </div>
</body>
</html>
"""

# -------------------------------
# 🔹 ROUTE
# -------------------------------

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""

    if request.method == 'POST':
        income = int(request.form['income'])
        loan = int(request.form['loan'])
        credit = int(request.form['credit'])
        employment = int(request.form['employment'])

        # 🔹 Same feature engineering for input
        ratio = loan / income

        input_data = [[income, loan, credit, employment, ratio]]
        input_data = scaler.transform(input_data)

        prediction = model.predict(input_data)

        # 🔥 SAFETY RULE (IMPORTANT)
        if credit >= 700 and income > loan and employment == 1:
            result = "✅ Loan Approved (High Confidence)"
        elif prediction[0] == 1:
            result = "✅ Loan Approved"
        else:
            result = "❌ Loan Rejected"

    return render_template_string(html, result=result)

# -------------------------------
# 🔹 RUN
# -------------------------------

if __name__ == '__main__':
    app.run(debug=True)
