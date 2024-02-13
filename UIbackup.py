import numpy as np
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mplcyberpunk


# Apply custom CSS styles for text color and background image
custom_css = """
<style>
      /* Change text color to white */
    .stText, .stHeader, .stCaption, h1, p {
        color: #FFFFFF !important; /* Set text color to white */
    }

     /* Change button color to black */
    .stButton>button {
        background-color: black !important;
        color: white !important; /* Set button label color to white */
    }

    /* Change input text color to black */
    .stTextInput>div>div>input {
        color: green !important;
    }
    /* Set background image */
    [data-testid="stAppViewContainer"] > .main {
        background-image: url("https://img.freepik.com/free-vector/geometric-neon-hexagonal-bipyramid-background-vector_53876-177932.jpg?size=626&ext=jpg&ga=GA1.2.190209625.1707203559&semt=ais");
        background-size: 200%;
        background-position: top left;
        background-repeat: no-repeat;
        background-attachment: local;
    }
</style>
"""

# Apply custom CSS styles
st.markdown(custom_css, unsafe_allow_html=True)


def get_interval(value):
    return max(1, int(value / 10))

# Introduction
st.title("Maria's Kitchen Sales Income Prediction")

st.write("""
Welcome to Maria's Kitchen Sales Income Prediction tool! As a thriving restaurant with a successful track record over the past 20 years, Maria's Kitchen is expanding to a new city.

To attract investors, we have developed this Sales Income Prediction application. It predicts the future sales income based on the sales records of the main restaurant for the last 10 years.

Our application allows you to input a specific year, and it will predict the potential sales income for that year. For example, you can predict the income for the 5th year of the new branch or the income for the 20th year, depending on your input.
""")

# User input
user_name = st.text_input("Enter name?")
x_label = st.text_input("Label for X axis?")
y_label = st.text_input("Label for Y axis?")
company_name = st.text_input("Branch name:")


# Data
years = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
sales_income = np.array([1, 2, 1.5, 3.3, 5.5, 4.4, 3.4, 3.8, 5, 4])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(years, sales_income, test_size=0.2, random_state=42)

# Ridge model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Prediction input with custom numeric input field
input_year_custom = st.number_input("Enter the year for prediction:", min_value=1, value=None)

if input_year_custom is not None and st.button("Generate Prediction"):
    # Display results
    st.write(f"Hello, {user_name}! This is your result:")
    st.write(f"Input {x_label}: {input_year_custom}")
    
    # Ridge Regression Prediction
    predicted_income = ridge_model.predict(np.array([[input_year_custom]]))[0]
    st.write(f"{x_label} {input_year_custom} Prediction: ${predicted_income:.2f} Million")

    # Description
    st.write("The predicted sales income represents an estimate based on the provided data and the Ridge Regression model.")

    # Accuracy
    test_score = ridge_model.score(X_test, y_test) * 100  # Convert accuracy to percentage
    st.write(f"Accuracy: {test_score:.2f}%")
    st.write("The accuracy is calculated based on the model's performance on the test data.")

    # Plot
    plt.style.use("cyberpunk")
    plt.scatter(X_train, y_train, color='white', label='Training Data', marker='o', s=30, edgecolor='black', linewidth=1.5)
    plt.scatter(X_test, y_test, color='gray', label='Test Data', marker='o', s=30, edgecolor='black', linewidth=1.5)

    years_extended_ridge = np.arange(min(years).item(), max(input_year_custom, 10) + 1, get_interval(input_year_custom)).reshape(-1, 1)
    plt.plot(years_extended_ridge, ridge_model.predict(years_extended_ridge), color='lime', label='Prediction Line', linewidth=1)

    # Connect all 10 data points with a line
    plt.plot(years, sales_income, color='cyan', label='Sales income history', linestyle='-', linewidth=1)

    if input_year_custom >= min(years):
        plt.scatter(input_year_custom, predicted_income, color='red', marker='o', label=f' {input_year_custom} year(s) Prediction', s=30, edgecolor='black', linewidth=1.5)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'Sales Income of {company_name} branch over the {x_label}s')
    plt.legend()

    all_years = np.arange(min(years).item(), max(input_year_custom, 10) + 1, get_interval(input_year_custom))
    plt.xticks(all_years)

    max_y_value = max(max(y_train), predicted_income)
    y_ticks = np.arange(0, max_y_value + 1, get_interval(max_y_value))
    plt.yticks(y_ticks)

    plt.grid(True, axis='both', linestyle='--', linewidth=0.5)

    st.pyplot(plt)
