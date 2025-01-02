# we need to import pandas in order to make it possible to do data wrangling 
import pandas as pd 
# I have to upload the CSV's so it is possible to do the data wranging. I began with the austin file so I can edit it. 
austin_housing= pd.read_csv('airbnb_listings_austin.csv')
austin_housing.info()
austin_housing.isnull().sum()
austin_df_sample = austin_housing.drop(columns=['square_feet','weekly_price','security_deposit','cleaning_fee','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','listing_url','description','summary','notes','transit','neighborhood_overview','host_response_time','host_response_rate', 'host_about', 'space' ])
austin_df_sample
austin_housing2 = austin_df_sample.dropna()
austin_housing2

# Check if the columns exist in the dataset
if 'price' in austin_housing2.columns:
    austin_housing2['price'] = austin_housing2['price'].replace('[\$,]', '', regex=True).astype(float)

if 'extra_people' in austin_housing2.columns:
    austin_housing2['extra_people'] = austin_housing2['extra_people'].replace('[\$,]', '', regex=True).astype(float)

# Verify the changes
print(austin_housing2[['price', 'extra_people']].head())
austin_housing2.to_csv('cleaned1_airbnb_listings.csv', index=False)

austin_housing2.info()

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df_clean = pd.read_csv('cleaned1_airbnb_listings.csv')

# Encode binary features
df_clean['host_is_superhost'] = df_clean['host_is_superhost'].apply(lambda x: 1 if x == 't' else 0)
df_clean['instant_bookable'] = df_clean['instant_bookable'].apply(lambda x: 1 if x == 't' else 0)

# Define features and target
categorical_features = ['property_type', 'room_type', 'cancellation_policy']
numerical_features = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 
                     'number_of_reviews', 'availability_365', 'host_listings_count']
binary_features = ['host_is_superhost', 'instant_bookable']
all_features = categorical_features + numerical_features + binary_features
target = 'price'

# Drop missing values
df_final = df_clean[all_features + [target]].dropna()

# Splitting data into features and target
X = df_final[all_features]
y = df_final[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: Scaling numerical and encoding categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Create a pipeline for preprocessing and modeling
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Predictions
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# Evaluate the model
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)

# Print results
print("Model Performance Metrics:")
print("--------------------------")
print(f"R-squared (Training): {r2_train:.3f}")
print(f"R-squared (Testing): {r2_test:.3f}")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")

# Get feature names after preprocessing
feature_names = (numerical_features + binary_features +
                [f"{feature}_{val}" for feature, vals in 
                 zip(categorical_features, 
                     pipeline.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .categories_) 
                 for val in vals[1:]])

# Get coefficients
coefficients = pipeline.named_steps['regressor'].coef_

# Create a coefficient plot for top 15 most influential features
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
coef_df['Abs_Coefficient'] = abs(coef_df['Coefficient'])
top_15_features = coef_df.nlargest(15, 'Abs_Coefficient')

plt.figure(figsize=(12, 8))
sns.barplot(x='Coefficient', y='Feature', data=top_15_features)
plt.title('Top 15 Most Influential Features in Linear Regression Model')
plt.tight_layout()
plt.show()

# Create actual vs predicted plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.tight_layout()
plt.show()

# Import necessary libraries and prepare the data
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df_clean = pd.read_csv('cleaned1_airbnb_listings.csv')

# Encode binary features
df_clean['host_is_superhost'] = df_clean['host_is_superhost'].apply(lambda x: 1 if x == 't' else 0)
df_clean['instant_bookable'] = df_clean['instant_bookable'].apply(lambda x: 1 if x == 't' else 0)

# Define features and target
categorical_features = ['property_type', 'room_type', 'cancellation_policy']
numerical_features = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 
                     'number_of_reviews', 'availability_365', 'host_listings_count']
binary_features = ['host_is_superhost', 'instant_bookable']
all_features = categorical_features + numerical_features + binary_features
target = 'price'

# Drop missing values
df_final = df_clean[all_features + [target]].dropna()

# Preprocessing: Scaling numerical and encoding categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Splitting data
X = df_final[all_features]
y = df_final[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform the data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Get feature names after preprocessing
feature_names = (numerical_features + binary_features +
                [f"{feature}_{val}" for feature, vals in 
                 zip(categorical_features, 
                     preprocessor.named_transformers_['cat'].categories_) 
                 for val in vals[1:]])

# Initialize and fit Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train_transformed, y_train)

# Extract feature importances
feature_importances = rf_model.feature_importances_

# Create DataFrame of feature importances
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Get top 10 features
top_10_features = feature_importance_df.head(10)

# Display results
print("\
Top 10 Most Important Features:")
print("--------------------------------")
print(top_10_features)

# Create visualization
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=top_10_features)
plt.title('Top 10 Most Important Features in Random Forest Model')
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.show()

# Calculate model performance metrics
y_pred_train = rf_model.predict(X_train_transformed)
y_pred_test = rf_model.predict(X_test_transformed)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\
Random Forest Model Performance:")
print("--------------------------------")
print(f"R-squared (Training): {r2_train:.3f}")
print(f"R-squared (Testing): {r2_test:.3f}")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('cleaned1_airbnb_listings.csv')

# Box plots for room_type and property_type vs price
plt.figure(figsize=(15, 6))

# Room Type vs Price
plt.subplot(1, 2, 1)
sns.boxplot(x='room_type', y='price', data=df)
plt.xticks(rotation=45)
plt.title('Room Type vs Price')

# Property Type vs Price (Top 5 Most Common)
plt.subplot(1, 2, 2)
top_5_property_types = df['property_type'].value_counts().nlargest(5).index
df_filtered = df[df['property_type'].isin(top_5_property_types)]
sns.boxplot(x='property_type', y='price', data=df_filtered)
plt.xticks(rotation=45)
plt.title('Property Type vs Price (Top 5 Most Common)')

plt.tight_layout()
plt.show()

# Print average price by room type
print("Average Price by Room Type:")
print(df.groupby('room_type')['price'].mean().sort_values(ascending=False))

# Get the top 5 most common property types
top_5_property_types = df['property_type'].value_counts().nlargest(5).index

# Filter the dataset to include only the top 5 property types
df_filtered = df[df['property_type'].isin(top_5_property_types)]

# Calculate the average price for each of the top 5 property types
average_price_by_property_type = df_filtered.groupby('property_type')['price'].mean().sort_values(ascending=False)

# Print the average price by property type
print("Average Price by Property Type (Top 5 Most Common):")
print(average_price_by_property_type)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt

airbnb_data = pd.read_csv('cleaned1_airbnb_listings.csv')


# Define the target variable (booked = 1 if availability_90 < 40%, else 0)
airbnb_data['booked'] = (airbnb_data['availability_90'] < 40).astype(int)

# Select features for the logistic regression model
features = ['price', 'accommodates', 'bathrooms', 'bedrooms', 'number_of_reviews', 'host_listings_count']
target = 'booked'

# Drop rows with missing values in the selected columns
data = airbnb_data[features + [target]].dropna()

# Split the data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred = log_reg.predict(X_test_scaled)
y_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.2f}")

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt

# Define the target variable (booked = 1 if availability_90 < 40%, else 0)
airbnb_data['booked'] = (airbnb_data['availability_90'] < 40).astype(int)

# Select features for the decision tree model
features = ['price', 'accommodates', 'bathrooms', 'bedrooms', 'number_of_reviews', 'host_listings_count']
target = 'booked'

# Drop rows with missing values in the selected columns
data = airbnb_data[features + [target]].dropna()

# Split the data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree classifier
tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_clf.fit(X_train, y_train)

# Make predictions
y_pred = tree_clf.predict(X_test)
y_prob = tree_clf.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.2f}")

# Plot the decision tree with a wider figure
plt.figure(figsize=(50, 25))  # Adjust figsize for a wider and more readable tree
plot_tree(tree_clf, feature_names=features, class_names=['Not Booked', 'Booked'], filled=True, fontsize=10)
plt.title("Decision Tree")
plt.show()

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

// This project analyzes Airbnb listings in Austin, Texas, using Python to uncover key factors influencing property pricing. The dataset was cleaned by removing unnecessary columns, ensuring proper data types, and verifying data integrity. A linear regression model was built using features such as the number of bedrooms, bathrooms, accommodates, and additional guest fees (extra_people) to identify significant predictors of price. Exploratory data analysis revealed strong correlations between property features and pricing, providing actionable insights for optimizing Airbnb listings. Implemented in Python, this project leverages libraries like pandas, numpy, matplotlib, and statsmodels, with clear visualizations and statistical summaries offering a comprehensive understanding of the data. Future extensions could include location-based analysis and advanced machine learning models to further refine pricing predictions.
