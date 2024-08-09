This project demonstrates the process of preprocessing categorical data for machine learning by performing one-hot encoding using Python's pandas and scikit-learn libraries.

Dependencies:

pandas
numpy
scikit-learn

Data:

The code assumes a pandas DataFrame named df with at least one column containing categorical data.
Steps:

Import necessary libraries: pandas, numpy, and OneHotEncoder from scikit-learn.
Place the .html files in a separate folder names "templates"
Categorize the CSS file into a new folder named "static"
Identify categorical columns in the DataFrame (e.g., object_cols).
Create a OneHotEncoder instance with handle_unknown='ignore' and sparse_output=False.
Fit and transform the categorical columns using the OneHotEncoder.
Create a pandas DataFrame from the encoded data.
Set the index of the encoded DataFrame to match the original DataFrame.
Set the column names of the encoded DataFrame using get_feature_names_out().
Drop the original categorical columns from the DataFrame.
Concatenate the original DataFrame with the encoded DataFrame.
Output:
The final DataFrame contains the original numerical columns and the one-hot encoded categorical columns. The model successfully returns price of the house on the basis of the parameters given by the user.
