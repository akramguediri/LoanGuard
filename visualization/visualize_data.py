import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_risk_scores(X_test, y_test, y_prob):
    risk_score = pd.DataFrame(X_test, columns=['age', 'income', 'loan_amount', 'loan_term', 'credit_score'])
    risk_score['default'] = y_test.values
    risk_score['risk_score'] = y_prob

    plt.figure(figsize=(10, 6))
    sns.histplot(risk_score['risk_score'], kde=True)
    plt.title('Distribution of Risk Scores')
    plt.xlabel('Risk Score')
    plt.ylabel('Frequency')
    plt.show()

    sns.pairplot(risk_score, hue='default', vars=['age', 'income', 'loan_amount', 'credit_score', 'risk_score'])
    plt.show()

if __name__ == "__main__":
    from data.preprocess_data import load_and_preprocess_data
    from model.train_model import train_and_evaluate_model
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data('data/loan_applicants.csv')
    model, _, _ = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    visualize_risk_scores(X_test, y_test, y_prob)
