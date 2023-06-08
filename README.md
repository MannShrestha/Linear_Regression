# Linear_Regression

## Car Price Prediction

### 1.	Importing Packages:
- Numpy
- Pandas
- Matplot
- Seaborn
- Sklearn

### 2.	Data Inspection
- Data Shape
- Data Description
- Data Information
- Checking Duplicates
- Checking Null Value

### 3.	Exploratory Data Analysis ( EDA )
Exploratory Data Analysis (EDA) is the process of examining and analyzing a dataset to understand its main characteristics, identify patterns, uncover relationships between variables, and gain insights.

#### **a. Univariate Analysis:**
*Univariate analysis is a statistical analysis technique that focuses on examining and understanding a single variable in isolation. It involves analyzing and summarizing the characteristics of that variable to gain insights into its distribution, central tendency, variability, and other relevant properties.*

***Techniques: Descriptive Statistics, Histograms, Box Plots, Probability Density Plots, Bar Charts, Pie Charts, Frequency Tables, Measures of Central Tendency, Measures of Dispersion, Percentiles***

#### **b. Bivariate Analysis:**
*Bivariate analysis is a statistical analysis technique that focuses on understanding the relationship between two variables. It involves examining the distribution, correlation, and association between two variables to determine the nature and strength of their relationship.*

***Methods/Technique used in bivariate analysis:
Scatter Plots, Correlation Analysis, Covariance, Cross-tabulation (Contingency Table), Chi-Square Test, T-tests/ANOVA, Regression Analysis***

### 4. Linaer Regression Analysis
*Linear regression analysis is a statistical technique used to model the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between the variables, meaning that changes in the independent variables are expected to have a constant effect on the dependent variable.*

*The goal of linear regression analysis is to find the best-fitting line that minimizes the differences between the observed data points and the predicted values based on the model. This line is often referred to as the regression line or the best-fit line.*

*The equation for a simple linear regression model with one independent variable is typically represented as:*
```
Y = β0 + β1*X + ε

Where:

    Y represents the dependent variable (the variable being predicted)
    X represents the independent variable (the variable used to predict Y)
    β0 represents the intercept (the value of Y when X is 0)
    β1 represents the slope (the change in Y for a unit change in X)
    ε represents the error term (the difference between the observed Y and the predicted Y)
```
The coefficients β0 and β1 are estimated using a method called ordinary least squares (OLS), which minimizes the sum of the squared differences between the observed and predicted values. The OLS method finds the values of β0 and β1 that result in the smallest overall error.

### 5. Visualising the Data
**Visualising Numeric Variables**
- Making a pairplot of all the numeric variables

### 6. Data Preparation
*Data preparation, also known as data preprocessing or data cleaning, is an essential step in the data analysis process. It involves transforming raw data into a format that is suitable for analysis. Data preparation helps to ensure the accuracy, consistency, and reliability of the data before applying statistical techniques or machine learning algorithms.*

**steps involved in data preparation:**
- Data Cleaning: This step involves handling missing values, removing duplicates, and correcting any errors or inconsistencies in the data. Missing values can be imputed using techniques like mean imputation or regression imputation. Duplicates can be identified and removed based on specific criteria. Errors or inconsistencies can be corrected by reviewing the data or applying validation rules.
- Data Transformation: Data transformation involves converting the data into a more suitable format for analysis. This may include scaling numerical variables to a common range, normalizing variables, or applying mathematical functions like logarithmic or exponential transformations. Data transformation can help to improve the distributional properties of the data and ensure that variables are on a similar scale.
- Variable Selection: If the dataset contains a large number of variables, it may be necessary to select a subset of relevant variables for analysis. This can be done based on domain knowledge, statistical measures such as correlation coefficients, or automated feature selection techniques. Removing irrelevant or redundant variables can improve computational efficiency and reduce the risk of overfitting in predictive modeling tasks.
- Handling Categorical Variables: Categorical variables need to be appropriately encoded for analysis. This can involve converting them into numerical values through techniques like one-hot encoding, label encoding, or ordinal encoding. The choice of encoding method depends on the nature of the categorical variable and the requirements of the analysis.
- Data Integration: In some cases, data from multiple sources or databases may need to be integrated or merged into a single dataset. This can involve matching and combining records based on common identifiers or creating new variables that consolidate information from different sources.
- Outlier Treatment: Outliers are extreme values that deviate significantly from the rest of the data. They can adversely affect analysis results or modeling accuracy. Outliers can be identified through statistical techniques such as the z-score, box plots, or domain knowledge. They can be handled by either removing them, transforming them, or replacing them with more reasonable values based on the context.
- Data Formatting: This may include setting the appropriate data types for variables, renaming variables for clarity, and ensuring consistent and meaningful variable labels.

### 7. Dummy Variables
***Encoding Technique --> Dummy Variable Encoding. 
Categorical Variables are converted into Neumerical Variables with the help of Dummy Variable Encoding***

### 8. Model Building
***Linear Regression Model***
* Splitting the Data into Training and Testing sets.

### 9. Rescaling the Features
Rescaling or feature scaling is a common preprocessing step in data analysis and machine learning. It involves transforming the numerical features in a dataset to a consistent scale. 

**Rescaling is performed for several reasons:**

- Comparability: Rescaling features ensures that variables with different scales and units are on a similar range. When variables have vastly different scales, it can lead to biased or misleading interpretations. Rescaling brings all features to a common scale, making them directly comparable and preventing one variable from dominating the analysis due to its larger magnitude.
- Algorithm Performance: Many machine learning algorithms are sensitive to the scale of the input features. Algorithms such as k-nearest neighbors (KNN), support vector machines (SVM), and gradient descent-based optimization methods converge faster and perform better when the features are on a similar scale. Rescaling helps to improve algorithm convergence, stability, and overall performance.
- Regularization: Regularization techniques, such as Ridge regression and Lasso regression, penalize large coefficients in a model. When the features have different scales, the regularization penalty may disproportionately affect variables with larger magnitudes. Rescaling the features ensures that regularization is applied fairly across all variables, preventing bias towards certain features.
- Interpretability: Rescaling features can aid in the interpretability of models and coefficients. When features are on a common scale, the coefficients associated with each feature can be directly compared to assess their relative importance or contribution to the model's predictions.


**There are different approaches to rescaling features. Here are two commonly used techniques:**

- [X] Min-Max Scaling (Normalization): 
- This method scales the features to a specified range, usually between 0 and 1. The formula for min-max scaling is: scaled_value = (value - min_value) / (max_value - min_value)
- [ ] Standardization (Z-score normalization): 
- This method transforms the features to have zero mean and unit variance. The formula for standardization is: scaled_value = (value - mean) / standard_deviation.


The choice between normalization and standardization depends on the specific requirements of the analysis and the characteristics of the data.

***In summary, rescaling features is important for comparability, algorithm performance, regularization, and interpretability. It ensures that the features are on a similar scale, preventing bias, improving performance, and facilitating meaningful comparisons between variables.***


### 10. Dividing into X and Y sets for the model building

### 11. RFE:

Recursive Feature Elimination (RFE) is a feature selection technique used in machine learning to select the most informative features from a given dataset. It is a backward elimination approach that starts with all features and iteratively removes the least important ones.

**overview of the RFE process:**

- Select a machine learning model: Choose a model that can assign importance scores to features, such as decision trees, linear regression, or random forests.
- Fit the model: Train the model using all the features in the dataset.
- Rank the features: Determine the importance of each feature based on the model's weights, coefficients, or other relevant measures.
- Eliminate features: Remove the least important feature(s) from the dataset.
- Repeat steps 2-4: Fit the model again using the remaining features, rank their importance, and eliminate the least important ones iteratively until a desired number of features remains.
- Evaluate performance: Assess the performance of the model using the selected features, such as accuracy, precision, or any relevant evaluation metric.

***RFE helps to identify and retain the most relevant features by iteratively removing the least significant ones. This process can reduce the dimensionality of the dataset, improve model interpretability, and potentially enhance model performance.***

It's important to note that the effectiveness of RFE depends on the choice of the machine learning model and the scoring method used to determine feature importance. Additionally, the number of features to select is a parameter that can be adjusted based on the specific problem and dataset.


### 12. Building model using statsmodel, for the detailed statistics

### 13. OLS regression, or ordinary least squares regression:

OLS regression, or ordinary least squares regression, is a statistical method used to estimate the relationship between a dependent variable and one or more independent variables. It is a widely used technique in econometrics and other fields of research.

***The goal of OLS regression is to find the best-fitting line, known as the regression line, that minimizes the sum of squared differences between the observed values of the dependent variable and the predicted values based on the independent variables.***

```
**The basic OLS regression model can be represented as:**
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ + ε
Where:
•	Y is the dependent variable (also called the response or outcome variable),
•	X₁, X₂, ..., Xₖ are the independent variables (also called predictors, explanatory variables, or regressors),
•	β₀, β₁, β₂, ..., βₖ are the regression coefficients (the parameters to be estimated), and
•	ε is the error term (representing the unexplained variation or random error).
The regression coefficients (β₀, β₁, β₂, ..., βₖ) quantify the relationship between the dependent variable and each independent variable. 
The intercept term (β₀) represents the value of the dependent variable when all independent variables are zero. The coefficients (β₁, β₂, ..., βₖ) indicate the change in the dependent variable associated with a one-unit change in each respective independent variable, assuming that other variables remain constant.
```

The OLS regression model estimates the values of the regression coefficients by minimizing the sum of squared residuals, which are the differences between the observed values of the dependent variable and the predicted values based on the regression equation. The estimation process involves finding the values of the coefficients that make the sum of squared residuals as small as possible.

Once the coefficients are estimated, various statistical tests and measures can be used to assess the overall fit of the model, the significance of individual coefficients, and the predictive power of the regression equation. Some commonly used measures include the coefficient of determination (R²), t-tests for coefficient significance, analysis of variance (ANOVA), and others.

OLS regression assumes several assumptions, such as linearity, independence of errors, homoscedasticity (constant variance of errors), and normality of errors. Violations of these assumptions may affect the reliability of the regression results, and additional techniques or robust regression methods may be necessary in such cases.

OLS regression is a versatile and widely used tool for analyzing the relationship between variables, making predictions, and testing hypotheses in many areas of research.

### 14. Variance Inflation Factor (VIF):
The Variance Inflation Factor (VIF) is a measure that quantifies the extent of multicollinearity in a regression model. Multicollinearity occurs when two or more independent variables in a regression model are highly correlated with each other. In such cases, it becomes difficult to determine the individual effects of these variables on the dependent variable.

The VIF is calculated for each independent variable in the model and measures the inflation in the variance of the estimated regression coefficients due to multicollinearity. Specifically, the VIF of an independent variable Xᵢ is calculated as:
```
VIFᵢ = 1 / (1 - Rᵢ²)
Where Rᵢ² is the coefficient of determination (R-squared) obtained by regressing Xᵢ on all other independent variables in the model. The VIF value quantifies how much the variance of the estimated coefficient of Xᵢ is inflated due to multicollinearity.
Interpreting VIF values:
•	VIF = 1: There is no multicollinearity. The independent variable is not correlated with any other variables in the model.
•	VIF > 1 and < 5: The independent variable has moderate multicollinearity. It is correlated with other variables but not to a problematic extent.
•	VIF ≥ 5: The independent variable has high multicollinearity. It is highly correlated with other variables, and its coefficient estimates may be unreliable.
```

In practice, a common rule of thumb is to consider independent variables with VIF values greater than 5 (or sometimes 10) as having high multicollinearity and potentially problematic. In such cases, it may be necessary to address multicollinearity by taking actions such as removing one of the correlated variables, transforming variables, or collecting additional data to reduce multicollinearity.

The VIF is a useful diagnostic tool to assess multicollinearity in regression models. By examining the VIF values, researchers can identify which independent variables are contributing to multicollinearity and take appropriate steps to address it, thereby improving the reliability and interpretability of the regression results.

### 15. Residual Analysis of the train data:
check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of the error terms and see what it looks like.

### 16. Making Predictions:
Applying the scaling on the test sets

### 17. Model Evaluation
