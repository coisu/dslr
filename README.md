# dslr
Discover Data Science through this project by recreating the Hogwarts Sorting Hat using logistic regression!

## Task Sequence
### Step 1: Data Analysis (describe.[extension])
> Objective: Calculate and output statistical summary information for the dataset.
    
    - Tasks:

        - Read the dataset_train.csv file.
        - For each numerical feature, compute the following:
            - Count of data points, mean, standard deviation, minimum/maximum values, and quartiles.
        - Present the results in table format.

Data Analysis (describe.py) Code Requirements
> The describe.py script analyzes numerical data and calculates the following information:

    - Count: The number of data points.
    - Mean: The average of the data.
    - Standard Deviation (Std): A measure of data dispersion.
    - Minimum (Min): The smallest value in the data.
    - Maximum (Max): The largest value in the data.
    - Percentiles: The 25th, 50th (median), and 75th percentiles.

Restrictions:

    - Using Pandas' describe() function or similar built-in - functions is strictly prohibited.
    - All calculations must be implemented manually.

### Step 2: Data Visualization
> Objective: Visually explore the data to understand relationships and characteristics of the features.
    - Tasks:

        - Histogram:
            - Visualize the distribution of scores for specific subjects across each house.
        - Scatter Plot:
            - Compare similar pairs of features.
        - Pair Plot:
            - Display relationships between multiple features simultaneously.

### Step 3: Logistic Regression
> Objective: Use logistic regression to classify students into their respective houses.
    - Tasks:

        - Model Training (logreg_train):
            - Train the model using the training dataset.
            Optimize weights using gradient descent.
        - Prediction (logreg_predict):
            - Predict the house for each student in the test dataset.
            Save the results in houses.csv format.

### Step 4: Review and Submission
> Objective: Test all functionalities to ensure proper operation and prepare for submission.

