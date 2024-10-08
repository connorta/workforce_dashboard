import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mesa import Agent, Model
from mesa.time import RandomActivation

# Title of the dashboard
st.title('Synthetic Workforce Modeling Dashboard')
st.markdown("""
This dashboard simulates workforce demographics, models retirement risk, and predicts attrition for the energy sector.
Use the controls below to generate and analyze synthetic workforce data or upload your own.
""")

# Sidebar options for user input
st.sidebar.header("Simulation Settings")

# Number of employees slider
num_employees = st.sidebar.slider("Number of Employees", min_value=50, max_value=500, value=100)

# Age range input
age_min = st.sidebar.slider("Minimum Age", min_value=20, max_value=50, value=45)
age_max = st.sidebar.slider("Maximum Age", min_value=55, max_value=70, value=65)

# Generate Synthetic Workforce Data
ages = np.random.normal(loc=(age_min + age_max) / 2, scale=5, size=num_employees).astype(int)
ages = np.clip(ages, age_min, age_max)
experience = ages - np.random.randint(22, 32, size=num_employees)
retirement_age = 60
retirement_prob = np.where(ages >= retirement_age, 0.8, 0.1)
skills = np.random.choice(['Basic', 'Intermediate', 'Advanced'], size=num_employees, p=[0.2, 0.5, 0.3])

# Create DataFrame
data = pd.DataFrame({
    'Employee_ID': range(1, num_employees + 1),
    'Age': ages,
    'Years_of_Experience': experience,
    'Retirement_Probability': retirement_prob,
    'Skill_Level': skills
})

# File Upload for Real Data
st.sidebar.subheader("Upload Your Workforce Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    real_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Workforce Data")
    st.write(real_data)
    data = real_data
else:
    st.write("Using synthetic data for demonstration.")

# Display the generated or uploaded data
st.subheader("Generated or Uploaded Workforce Data")
st.write(data)

# Retirement Simulation
st.subheader("Retirement Simulation Over 5 Years")
for year in range(1, 6):
    data[f'Retired_Year_{year}'] = np.random.binomial(1, data['Retirement_Probability'])
    data[f'Active_Year_{year}'] = 1 - data[f'Retired_Year_{year}']

# Display retirement trends
retirements = data[[f'Retired_Year_{year}' for year in range(1, 6)]].sum()
active_employees = [data[f'Active_Year_{year}'].sum() for year in range(1, 6)]

# Plot the number of active employees each year
fig, ax = plt.subplots()
ax.plot(range(1, 6), active_employees, marker='o', color='blue')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Active Employees')
ax.set_title('Projected Workforce Size Over 5 Years')
st.pyplot(fig)

# Predictive Attrition Modeling
def build_attrition_model(data):
    # Create synthetic features for the machine learning model
    data['Engagement_Score'] = np.random.randint(1, 10, data.shape[0])
    data['Performance_Rating'] = np.random.randint(1, 5, data.shape[0])
    data['Attrition'] = np.random.choice([0, 1], size=data.shape[0], p=[0.7, 0.3])

    # Prepare data for ML
    X = data[['Age', 'Years_of_Experience', 'Engagement_Score', 'Performance_Rating']]
    y = data['Attrition']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict attrition
    data['Attrition_Probability'] = model.predict_proba(X)[:, 1]
    return data

# Apply the attrition model to the data and display in Streamlit
data = build_attrition_model(data)
st.subheader("Attrition Prediction")
st.write(data[['Employee_ID', 'Age', 'Attrition_Probability']])

# Visualize Attrition Risk
st.subheader("Attrition Risk Visualization")
fig, ax = plt.subplots()
ax.hist(data['Attrition_Probability'], bins=10, color='orange')
ax.set_xlabel('Attrition Probability')
ax.set_ylabel('Number of Employees')
st.pyplot(fig)

# Agent-Based Modeling for Retirement Simulation
class EmployeeAgent(Agent):
    def __init__(self, unique_id, model, age, experience):
        super().__init__(unique_id, model)
        self.age = age
        self.experience = experience

    def step(self):
        if self.age >= 60:
            self.model.retirement_count += 1

class WorkforceModel(Model):
    def __init__(self, num_employees):
        self.num_employees = num_employees
        self.schedule = RandomActivation(self)

        # Add agents to the model
        for i in range(self.num_employees):
            age = np.random.randint(45, 65)
            experience = np.random.randint(10, 40)
            agent = EmployeeAgent(i, self, age, experience)
            self.schedule.add(agent)

    def step(self):
        self.retirement_count = 0
        self.schedule.step()

st.subheader("Agent-Based Retirement Simulation")
model = WorkforceModel(num_employees)
for i in range(5):  # Run the model for 5 steps (years)
    model.step()
st.write(f"Total retirements after 5 years: {model.retirement_count}")

# Skill Gap Analysis and Training Recommendations
training_courses = {
    'Data Analysis': 'Coursera: Data Analysis for Business',
    'Python Programming': 'Udemy: Python for Beginners',
    # Additional courses...
}

st.sidebar.subheader("Employee Skill Gap Analysis")
skill_to_learn = st.sidebar.selectbox("Select a skill to analyze", list(training_courses.keys()))
st.sidebar.write(f"Suggested Training: {training_courses[skill_to_learn]}")

# Career Path Visualization
st.subheader("Career Path Simulation")
career_paths = {
    'Entry Level': ['Junior Analyst', 'Data Analyst', 'Senior Analyst'],
    'Technical Role': ['Junior Developer', 'Software Engineer', 'Lead Engineer']
}

selected_path = st.selectbox("Select a Career Path", list(career_paths.keys()))
st.write(" -> ".join(career_paths[selected_path]))

# Industry Benchmarking Report
st.subheader("Industry Benchmarking")
if uploaded_file is not None:
    insights = data.groupby('Skill_Level').agg({'Age': 'mean', 'Years_of_Experience': 'mean'})
    st.write("Average Age and Experience by Skill Level")
    st.write(insights)

    # Download button for benchmark report
    st.download_button(
        label="Download Benchmark Report",
        data=insights.to_csv(),
        file_name='benchmark_report.csv'
    )
