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


