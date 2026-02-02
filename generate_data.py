import pandas as pd
import numpy as np

def generate_ethical_dataset(n=2000):
    np.random.seed(42)
    
    data = {
        'Age': np.random.randint(18, 75, n),
        'Gender': np.random.choice(['Male', 'Female', 'Non-Binary'], n),
        'Race': np.random.choice(['Group_A', 'Group_B', 'Group_C'], n),
        'Income': np.random.randint(20, 150, n) * 1000,
        'Credit_Score': np.random.randint(300, 850, n),
        'Employment_Years': np.random.randint(0, 40, n)
    }
    
    df = pd.DataFrame(data)
    
    # Simple logic: Approval is based on Credit and Income
    # But we add "noise" so the AI might accidentally find patterns in Race/Gender
    score = (df['Credit_Score'] / 850) * 0.5 + (df['Income'] / 150000) * 0.4 + (df['Employment_Years'] / 40) * 0.1
    df['Loan_Status'] = (score > 0.5).astype(int)
    
    df.to_csv('loan_experiment_data.csv', index=False)
    print("Dataset Created: loan_experiment_data.csv")

if __name__ == "__main__":
    generate_ethical_dataset()