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

- Data Analysis (describe.py) Code Requirements
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
        - displays a histogram answering the next question:
          
            _"Which Hogwarts course has a homogeneous score distribution between all four houses?"_
         
            ![std_line_graph](https://github.com/user-attachments/assets/c571156f-084e-47ee-b90a-bf12597922b9)
            ![Charms_histogram](https://github.com/user-attachments/assets/6a0972ab-ca69-4b66-a489-7cf27f793516)
            ![History of Magic_histogram](https://github.com/user-attachments/assets/6de17b98-aeb9-4e33-bde9-efe5fba5f0b7)
            ![Potions_histogram](https://github.com/user-attachments/assets/fdfcd5f6-4dc0-41df-ae16-cd08f71880c4)
            ![Care of Magical Creatures_histogram](https://github.com/user-attachments/assets/faa0cb86-07a2-47d0-aef2-1e90be906018)
    - Scatter Plot:
        - displays a scatter plot answering the next question :
          
            _"What are the two features that are similar ?"_
          
    - Pair Plot:
        - Display relationships between multiple features simultaneously.
        - displays a pair plot or scatter plot matrix (according to the library that you are using).
            > From this visualization, what features are you going to use for your logistic regression?

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



## 🛠️ Tech Stack

### Programming Languages
- **Python**: Core programming language used for data analysis and visualization.
- **Bash**: Automating processes and task execution.

### Data Analysis & Visualization
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For creating static, interactive, and animated visualizations.
- **Seaborn**: For statistical data visualization.

### Machine Learning (Optional, if applicable)
- **Scikit-learn**: For machine learning models and evaluation.
- **NumPy**: For numerical computing and array manipulation.

### Development Tools
- **Makefile**: For task automation (e.g., cleaning, building, and running scripts).
- **Docker** (Optional): Containerized environment for consistent development and deployment.

### Project Management
- **Git**: Version control and collaboration.
- **GitHub**: Repository hosting and issue tracking.

### Other Tools
- **Jupyter Notebook**: Planning to implement

---


