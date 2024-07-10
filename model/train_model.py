from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return model, accuracy, report

if __name__ == "__main__":
    from data.preprocess_data import load_and_preprocess_data
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data('data/loan_applicants.csv')
    
    model, accuracy, report = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)
