import pandas as pd
import numpy as np
import os

def generate_data(n=1000, seed=123):
    np.random.seed(seed)
    data = []

    for _ in range(n):
        shoulders = np.random.uniform(35, 45)
        bust = np.random.uniform(32, 44)
        waist = np.random.uniform(24, 36)
        hips = np.random.uniform(34, 46)
        whr = waist / hips

        if shoulders < bust < hips and whr < 0.75:
            shape = 'Pear'
        elif shoulders > bust > hips and whr >= 0.75:
            shape = 'Inverted Triangle'
        elif abs(bust - hips) <= 2 and (bust - waist) >= 8 and whr <= 0.7:
            shape = 'Hourglass'
        elif bust > waist and abs(waist - hips) <= 2 and whr > 0.85:
            shape = 'Apple'
        elif abs(bust - waist) <= 2 and abs(waist - hips) <= 2 and 0.75 <= whr <= 0.8:
            shape = 'Rectangle'
        else:
            shape = 'Other'  # For outliers or unclassifiable shapes

        data.append([shoulders, bust, waist, hips, whr, shape])

    df = pd.DataFrame(data, columns=['Shoulders', 'Bust', 'Waist', 'Hips', 'WHR', 'BodyShape'])
    df = df[df['BodyShape'] != 'Other']  
    return df

if __name__ == '__main__':
    df = generate_data()

    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    data_dir = os.path.join(parent_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    output_path = os.path.join(data_dir, 'synthetic_body_data.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Save synthetic data: {output_path}")
