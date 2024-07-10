import argparse
import pandas as pd
from data.generate_data import generate_data
from data.preprocess_data import load_and_preprocess_data
from model.train_model import train_and_evaluate_model
from visualization.visualize_data import visualize_risk_scores

def fetch_applicant_data():
    # Simulating an API call
    applicant_data = {
        'age': 35,
        'income': 75000,
        'loan_amount': 30000,
        'loan_term': 36,
        'credit_score': 680
    }
    return pd.DataFrame([applicant_data])

def main():
    parser = argparse.ArgumentParser(description='Credit Risk Analysis Tool')
    parser.add_argument('--generate', '-g', action='store_true', help='Generate new data and process it')
    parser.add_argument('--train', '-t', action='store_true', help='Train the model')
    parser.add_argument('--plot', '-p', action='store_true', help='Visualize the data')
    parser.add_argument('--fetch', '-f', action='store_true', help='Fetch and score a new applicant')

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    if args.generate:
        print("Generating and processing data...")
        generate_data(1000, 'data/loan_applicants.csv')
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('data/loan_applicants.csv')
        print("Data generation and processing completed.")

    if args.train:
        print("Training the model...")
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('data/loan_applicants.csv')
        model, accuracy, report = train_and_evaluate_model(X_train, X_test, y_train, y_test)
        print(f'Accuracy: {accuracy}')
        print('Classification Report:')
        print(report)

    if args.plot:
        print("Visualizing the data...")
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('data/loan_applicants.csv')
        model, _, _ = train_and_evaluate_model(X_train, X_test, y_train, y_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        visualize_risk_scores(X_test, y_test, y_prob)

    if args.fetch:
        print("Fetching and scoring a new applicant...")
        new_applicant_data = {}
        if args.fetch:
            new_applicant_data['age'] = int(input("Enter age: "))
            new_applicant_data['income'] = int(input("Enter income: "))
            new_applicant_data['loan_amount'] = int(input("Enter loan amount: "))
            new_applicant_data['loan_term'] = int(input("Enter loan term: "))
            new_applicant_data['credit_score'] = int(input("Enter credit score: "))
        else:
            new_applicant_data = {
                'age': 35,
                'income': 75000,
                'loan_amount': 30000,
                'loan_term': 36,
                'credit_score': 680
            }

        new_applicant = pd.DataFrame([new_applicant_data])
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data('data/loan_applicants.csv')
        model, _, _ = train_and_evaluate_model(X_train, X_test, y_train, y_test)
        new_applicant_scaled = scaler.transform(new_applicant)
        new_applicant_risk_score = model.predict_proba(new_applicant_scaled)[:, 1]
        print(f'Risk score for new applicant: {new_applicant_risk_score[0]}')

if __name__ == "__main__":
    main()