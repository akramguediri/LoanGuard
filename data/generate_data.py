import pandas as pd
from faker import Faker
import random

def generate_data(num_records, file_path):
    fake = Faker()
    data = []
    for _ in range(num_records):
        age = random.randint(18, 70)
        income = random.randint(20000, 150000)
        loan_amount = random.randint(5000, 50000)
        loan_term = random.choice([12, 24, 36, 48, 60])
        credit_score = random.randint(300, 850)
        default = random.choice([0, 1])
        data.append([age, income, loan_amount, loan_term, credit_score, default])
    
    columns = ['age', 'income', 'loan_amount', 'loan_term', 'credit_score', 'default']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    generate_data(1000, 'data/loan_applicants.csv')
