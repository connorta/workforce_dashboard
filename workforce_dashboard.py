import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mesa import Agent, Model
from mesa.time import RandomActivation
import openai

# Title of the dashboard
st.title('Advanced Workforce Modeling Dashboard with Machine Learning')
st.markdown("""
This dashboard simulates workforce demographics, models retirement and attrition risks, and provides predictive insights for workforce planning in the energy sector.
Use the controls below to generate and analyze synthetic workforce data or upload your own.
""")

# Sidebar options for user input
st.sidebar.header("Simulation Settings")

# Number of employees slider
num_employees = st.sidebar.slider("Number of Employees", min_value=50, max_value=1000, value=200)

# Age range input
age_min = st.sidebar.slider("Minimum Age", min_value=20, max_value=50, value=30)
age_max = st.sidebar.slider("Maximum Age", min_value=55, max_value=70, value=65)

# Generate Synthetic Workforce Data
np.random.seed(42)
ages = np.random.normal(loc=(age_min + age_max) / 2, scale=5, size=num_employees).astype(int)
ages = np.clip(ages, age_min, age_max)
experience = np.clip(ages - np.random.randint(22, 32, size=num_employees), 0, None)
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

# Feature Engineering
skill_mapping = {'Basic': 0, 'Intermediate': 1, 'Advanced': 2}
data['Skill_Level_Code'] = data['Skill_Level'].map(skill_mapping)
data['Engagement_Score'] = np.random.randint(1, 11, size=num_employees)
data['Performance_Rating'] = np.random.randint(1, 6, size=num_employees)
data['Attrition'] = np.random.choice([0, 1], size=num_employees, p=[0.8, 0.2])

# Predictive Attrition Modeling
def build_attrition_model(data):
    # Prepare data for ML
    X = data[['Age', 'Years_of_Experience', 'Skill_Level_Code', 'Engagement_Score', 'Performance_Rating']]
    y = data['Attrition']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Gradient Boosting Classifier
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict attrition
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    # Predict attrition probability for all employees
    data['Attrition_Probability'] = model.predict_proba(X)[:, 1]
    return data, accuracy, report, confusion

# Apply the attrition model to the data and display in Streamlit
data, accuracy, report, confusion = build_attrition_model(data)
st.subheader("Attrition Prediction Model Results")
st.write(f"Model Accuracy: {accuracy:.2f}")
st.text("Classification Report:")
st.text(report)
st.text("Confusion Matrix:")
st.write(confusion)

# Visualize Attrition Risk
st.subheader("Attrition Risk Visualization")
fig, ax = plt.subplots()
ax.hist(data['Attrition_Probability'], bins=10, color='orange')
ax.set_xlabel('Attrition Probability')
ax.set_ylabel('Number of Employees')
ax.set_title('Distribution of Attrition Probability')
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
    'Leadership': 'LinkedIn Learning: Leadership Foundations',
    'Project Management': 'edX: Introduction to Project Management'
}

st.sidebar.subheader("Employee Skill Gap Analysis")
skill_to_learn = st.sidebar.selectbox("Select a skill to analyze", list(training_courses.keys()))
st.sidebar.write(f"Suggested Training: {training_courses[skill_to_learn]}")

# Career Path Visualization
st.subheader("Career Path Simulation")
career_paths = {
    'Entry Level': ['Junior Analyst', 'Data Analyst', 'Senior Analyst'],
    'Technical Role': ['Junior Developer', 'Software Engineer', 'Lead Engineer'],
    'Management': ['Team Lead', 'Project Manager', 'Director']
}

selected_path = st.selectbox("Select a Career Path", list(career_paths.keys()))
st.write(" -> ".join(career_paths[selected_path]))

# Industry Benchmarking Report
st.subheader("Industry Benchmarking")
if uploaded_file is not None:
    insights = data.groupby('Skill_Level').agg({'Age': 'mean', 'Years_of_Experience': 'mean', 'Engagement_Score': 'mean'})
    st.write("Average Age, Experience, and Engagement Score by Skill Level")
    st.write(insights)

    # Download button for benchmark report
    st.download_button(
        label="Download Benchmark Report",
        data=insights.to_csv(),
        file_name='benchmark_report.csv'
    )

# Supply and Demand Analysis of Skills
st.subheader("Supply and Demand of Skills")
# Simulate skill demand based on external factors
demand = {'Basic': np.random.randint(30, 80), 'Intermediate': np.random.randint(100, 200), 'Advanced': np.random.randint(50, 100)}
skill_supply = data['Skill_Level'].value_counts().reindex(['Basic', 'Intermediate', 'Advanced'], fill_value=0).to_dict()

# Create DataFrame for supply and demand comparison
supply_demand_df = pd.DataFrame({'Supply': skill_supply, 'Demand': demand})

st.write("Skill Supply vs Demand")
st.write(supply_demand_df)

# Plot supply vs demand
fig, ax = plt.subplots()
supply_demand_df.plot(kind='bar', ax=ax)
ax.set_xlabel('Skill Level')
ax.set_ylabel('Count')
ax.set_title('Supply and Demand of Skills')
st.pyplot(fig)

# GPT Agent for Q&A
st.sidebar.header("Ask the GPT Agent")
user_question = st.sidebar.text_area("Ask a question about the analysis:")

if user_question:
    openai.api_key = st.secrets["openai_api_key"]
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"You are an expert data analyst. Answer the following question based on the workforce data analysis provided: {user_question}",
        max_tokens=150
    )
    st.sidebar.write("GPT Agent Response:")
    st.sidebar.write(response.choices[0].text.strip())
