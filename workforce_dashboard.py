import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Title of the dashboard
st.title('Synthetic Workforce Modeling Dashboard')
st.markdown("""
This dashboard simulates workforce demographics and models retirement risk for the energy sector.
Use the controls below to generate and analyze synthetic workforce data.
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

# Display the generated data
st.subheader("Generated Synthetic Workforce Data")
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

# Skill Level Distribution Over Time
st.subheader("Skill Level Distribution at Year 1")
skill_distribution = data['Skill_Level'].value_counts()
st.bar_chart(skill_distribution)

st.markdown("""
### Insights:
- The number of active employees decreases over time as retirements happen.
- The skill level distribution helps identify potential skill gaps emerging due to retirements.
""")
