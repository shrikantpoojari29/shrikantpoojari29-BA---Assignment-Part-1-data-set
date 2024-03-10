## **Step 1: Clean and Preprocess the Data**

First, we ensure the data is clean and ready for analysis.

```python
# Handle missing values
data.dropna(inplace=True)

# Encode categorical variables
data_encoded = pd.get_dummies(data)

# Scale numerical features if needed
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_encoded)


## **Step 2: Exploratory Analysis**

Explore the dataset to gain insights.

```python
# Explore distribution of the target variable
sns.countplot(x='Churn', data=data)

# Visualize the relationship between features and churn
sns.boxplot(x='Churn', y='Tenure', data=data)

# Analyze correlations
correlation_matrix = data_encoded.corr()
sns.heatmap(correlation_matrix, annot=True)

## Step 3: **Build a Classification Model**

Construct a classification model to predict churn.

```python
X = data_scaled.drop('Churn', axis=1)
y = data_scaled['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

## Step 4: **Summary of Factors Influencing Churn and Preventive Actions**

Summarize the factors influencing churn and suggest preventive actions.

```python
# Analyze feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.coef_[0]})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print(feature_importance)

# Provide actionable insights based on analysis
# E.g., Offer personalized discounts to customers with low satisfaction scores or long tenure.
# Improve product features based on feedback from customers who complained.
