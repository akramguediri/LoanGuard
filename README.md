# LoanGuard
Credit Risk Analysis Tool
```plaintext
            .                ..               _                           
                    .   .                    | |    ___   __ _ _ __       
            .       .   .    ..              | |   / _ \ / _` | '_ \      
            .   .   .   .  ......  .         | |__| (_) | (_| | | | |     
                        ..&&(..,&&..         |_____\___/ \__,_|_| |_|  _ 
             .......    .&.&,..%&.&.         / ___|_   _  __ _ _ __ __| |
           ..&&...%&&.  ..&&....&&..        | |  _| | | |/ _` | '__/ _` |
           .&.&...&....   ........          | |_| | |_| | (_| | | | (_| |
           ..&&...&&/.                       \____|\__,_|\__,_|_|  \__,_|
              ......              ..&&..
         ....................   ..&&&.. 
 .####...&&&&&&&&&&&&&&&&&&&...&&&&..   
 .####.&&&&&&&&&&&&&&&&&&&&&&&&&..      
 .####.&&&&&&&&&&&&&&&&&&&&&&&..        
 .####.&&....................           
 .####..
```
# Credit Risk Analysis Tool


## Description

The Credit Risk Analysis Tool is a Python-based application designed to assess the creditworthiness of loan applicants. It integrates data generation, machine learning model training (Logistic Regression), data visualization, and scoring of new applicants based on their provided data.

## Features

- **Data Generation:** Hypothetical data can be generated using the Faker library and saved to CSV files.
- **Model Training:** Utilizes Scikit-learn for training a Logistic Regression model to predict loan default risk.
- **Data Visualization:** Matplotlib and Seaborn are employed to visualize the distribution and relationships within the dataset.
- **Scoring New Applicants:** Allows users to input applicant data interactively or use default values to predict their credit risk score.

## Tech Stack

- Python
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn
- Faker (for generating synthetic data)

## Installation

1. **Clone the repository:**

   ```
   git clone https://github.com/akramguediri/LoanGuard.git
   cd credit-risk-analysis
   ```

2. **Install dependencies:**

   ```
   pip install -r requirements.txt
   ```

## Usage

### Generating Data

To generate new data and process it:

```
python main.py --generate
```

### Training the Model

To train the machine learning model:

```
python main.py --train
```

### Visualizing Data

To visualize the dataset:

```
python main.py --plot
```

### Scoring a New Applicant

To fetch and score a new applicant:

```
python main.py --fetch
```

If no applicant data is provided, default values will be used.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
