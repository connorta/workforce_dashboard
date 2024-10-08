import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import openai
import itertools

# Title of the dashboard
st.title('Workforce Skills Inventory and Project Alignment Dashboard')
st.markdown("""
This dashboard helps you understand the skills across your organization, aligns talent with project requirements, predicts changes in workforce, and creates actionable insights for effective workforce management.
Use the controls below to input skills data and upcoming projects, then optimize the allocation of resources.
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
skills = np.random.choice(['Python', 'Data Analysis', 'Project Management', 'Leadership', 'Cloud Computing'], size=num_employees, p=[0.25, 0.25, 0.2, 0.15, 0.15])
proficiency = np.random.choice(['Beginner', 'Intermediate', 'Advanced'], size=num_employees, p=[0.3, 0.5, 0.2])

# Create DataFrame
data = pd.DataFrame({
    'Employee_ID': range(1, num_employees + 1),
    'Age': ages,
    'Years_of_Experience': experience,
    'Skill': skills,
    'Proficiency_Level': proficiency
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
proficiency_mapping = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}
data['Proficiency_Code'] = data['Proficiency_Level'].map(proficiency_mapping)
data['Engagement_Score'] = np.random.randint(1, 11, size=num_employees)
data['Performance_Rating'] = np.random.randint(1, 6, size=num_employees)
data['Attrition'] = np.random.choice([0, 1], size=num_employees, p=[0.8, 0.2])

# Predictive Attrition Modeling
def build_attrition_model(data):
    # Prepare data for ML
    X = data[['Age', 'Years_of_Experience', 'Proficiency_Code', 'Engagement_Score', 'Performance_Rating']]
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

# Input Upcoming Project Requirements
st.sidebar.header("Project Requirements")
num_projects = st.sidebar.number_input("Number of Projects", min_value=1, max_value=20, value=3)
project_data = []

for i in range(num_projects):
    st.sidebar.subheader(f"Project {i + 1}")
    project_name = st.sidebar.text_input(f"Project {i + 1} Name", value=f"Project {i + 1}")
    required_skills = st.sidebar.multiselect(f"Skills Needed for {project_name}", ['Python', 'Data Analysis', 'Project Management', 'Leadership', 'Cloud Computing'])
    skill_capacities = st.sidebar.slider(f"Capacity Needed for {project_name} (Number of Employees)", min_value=1, max_value=50, value=5)
    project_data.append({'Project_Name': project_name, 'Skills_Needed': required_skills, 'Capacity_Needed': skill_capacities})

# Matching Employees to Projects
st.subheader("Project Allocation and Skill Matching")
matched_projects = []

for project in project_data:
    required_skills = project['Skills_Needed']
    capacity_needed = project['Capacity_Needed']
    available_employees = data[data['Skill'].isin(required_skills)]
    selected_employees = available_employees.head(capacity_needed)
    matched_projects.append({'Project': project['Project_Name'], 'Assigned_Employees': selected_employees})

    st.write(f"**{project['Project_Name']}**")
    st.write(selected_employees[['Employee_ID', 'Skill', 'Proficiency_Level', 'Years_of_Experience']])

# Skill Gap Analysis
st.subheader("Skill Gap Analysis")
all_required_skills = list(itertools.chain.from_iterable([project['Skills_Needed'] for project in project_data]))
required_skill_counts = pd.Series(all_required_skills).value_counts()
available_skill_counts = data['Skill'].value_counts()
skill_gap = required_skill_counts.subtract(available_skill_counts, fill_value=0)

st.write("Required vs Available Skills")
st.write(pd.DataFrame({'Required': required_skill_counts, 'Available': available_skill_counts, 'Gap': skill_gap}))

# GPT Agent for Q&A
st.sidebar.header("Ask the GPT Agent")
user_question = st.sidebar.text_area("Ask a question about the analysis:")

if user_question:
    openai.api_key = st.secrets["openai_api_key"]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert data analyst."},
            {"role": "user", "content": user_question}
        ],
        max_tokens=150
    )
    st.sidebar.write("GPT Agent Response:")
    st.sidebar.write(response['choices'][0]['message']['content'].strip())
