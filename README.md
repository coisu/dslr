# dslr
Discover Data Science through this project by recreating the Hogwarts Sorting Hat using logistic regression!

### Table of Contents
- [Task](#task-sequence)
- [Feature Scaling](#feature-scaling)
- [Tech Stack](#tech-stack)


## Usage
```bash
  git clone git@github.com:coisu/dslr.git
  cd dslr

  # Build Docker image
  make         
  # Show summary statistics for numeric features       
  make describe
  # Generate histograms
  make histogram
  # Scatter plots for top correlated feature pairs
  make scatter_plot  
  # Generate pair plots
  make pair_plot      
  # Train the Magic Hat
  make train          
  # The MAgic Hat assign students in Houses
  make magic_hat    
  # Accurancy evaluation  
  make eval
```

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

---

## Feature Scaling
## **Normalization and Standardization**

Normalization and Standardization are two key preprocessing techniques for scaling data. While they both adjust the scale of features, they serve different purposes and impact data differently. Here's a detailed comparison:

---

## **1. Definitions**

### **Normalization**
- **Definition**: Scales data to a fixed range, typically [0, 1].
- **Formula**:

![image](https://github.com/user-attachments/assets/8de435b3-7819-41f1-9e58-05c41edccee2)

- **Result**:
  - Transforms all feature values to lie between 0 and 1.
  - Retains the original distribution shape.

---

### **Standardization**
- **Definition**: Scales data to have a mean of 0 and a standard deviation of 1.
- **Formula**:

![image](https://github.com/user-attachments/assets/6ad11025-8725-4cf7-b3cc-bef5ecedb3f3)

- **Result**:
  - Transforms data to have a standard normal distribution.
  - Adjusts for centrality (mean) and spread (variance).

---

## **2. Key Differences**

| **Aspect**         | **Normalization**                          | **Standardization**                       |
|---------------------|--------------------------------------------|-------------------------------------------|
| **Purpose**         | Scales data to a specific range (0-1).     | Adjusts data to have mean 0, std. dev. 1. |
| **Impact on Data**  | Retains original distribution.             | Approximates a standard normal distribution. |
| **Result Range**    | [0, 1]                                     | Mean = 0, Std. Dev = 1                    |
| **Sensitivity to Outliers** | Very sensitive.                     | Less sensitive.                           |
| **Applications**    | Neural Networks, Fixed-scale requirements. | Logistic Regression, SVM, PCA, KNN.       |
| **Python Library**  | scikit-learn, MinMaxScaler                 | scikit-learn, StandardScaler                |
---

## **3. Examples**

### **Input Data**
| Feature1 | Feature2 | Feature3 |
|----------|----------|----------|
| 10       | 100      | -10      |
| 20       | 200      | -5       |
| 30       | 300      | 0        |
| 40       | 400      | 5        |
| 50       | 500      | 10       |

### **Normalized Data**
| Feature1 | Feature2 | Feature3 |
|----------|----------|----------|
| 0.00     | 0.00     | 0.00     |
| 0.25     | 0.25     | 0.25     |
| 0.50     | 0.50     | 0.50     |
| 0.75     | 0.75     | 0.75     |
| 1.00     | 1.00     | 1.00     |

### **Standardized Data**
| Feature1  | Feature2  | Feature3  |
|-----------|-----------|-----------|
| -1.41     | -1.41     | -1.41     |
| -0.71     | -0.71     | -0.71     |
|  0.00     |  0.00     |  0.00     |
|  0.71     |  0.71     |  0.71     |
|  1.41     |  1.41     |  1.41     |

---

## **4. Choosing the Right Technique**

### **When to Use Normalization**
- The feature's absolute values matter, and the scale needs to be fixed (e.g., [0, 1]).
- **Applications**:
  - Neural Networks (e.g., for stable training in input layers).
  - Data visualization where all features must be scaled proportionally.

### **When to Use Standardization**
- The feature's distribution is important (e.g., mean and variance).
- **Applications**:
  - Linear models like Logistic Regression, SVM, KNN.
  - PCA or other techniques sensitive to variance.

---

## **5. Practical Insights**

- **Outliers**:
  - Normalization is highly sensitive to outliers since it uses `min` and `max` values.
  - Standardization is less sensitive because it focuses on the mean and standard deviation.

- **Model Requirements**:
  - Many models (e.g., neural networks) require input data to be normalized.
  - Standardization is essential for models that rely on distances (e.g., KNN, SVM).

---

### **Conclusion**

- Use **Normalization** when all features need to be scaled to a fixed range (e.g., 0-1).
- Use **Standardization** when the distribution of data (mean and variance) is important, especially for distance-based or linear models.

Both techniques have their use cases, and the choice depends on the specific problem and model requirements. 😊





---

🛠️ 
## Tech Stack

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





