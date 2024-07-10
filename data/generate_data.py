import pandas as pd
from faker import Faker
import random

def generate_data(num_records, file_path):
    fake = Faker()
    data = []
    for _ in range(num_records):
        age = random.randint(18, 70)
        income = random.randint(24000, 250000)
        loan_amount = random.randint(5000, 500000)
        loan_term = random.choice([12, 24, 36, 48, 60])
        credit_score = random.randint(700, 850)
        default = int((income * (loan_term / 12)) * 0.8 >= (loan_amount * 1.2) and credit_score >= 750)
        data.append([age, income, loan_amount, loan_term, credit_score, default])
    
    columns = ['age', 'income', 'loan_amount', 'loan_term', 'credit_score', 'default']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    generate_data(1000, 'data/loan_applicants.csv')
