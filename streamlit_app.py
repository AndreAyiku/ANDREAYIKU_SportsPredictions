#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model_path = 'bestmodel.pkl'
model = joblib.load(model_path)

# Define the columns expected by the model
expected_columns = [
    'movement_reactions', 'potential', 'passing', 'mentality_composure', 'value_eur',
    'dribbling', 'attacking_short_passing', 'mentality_vision', 'international_reputation',
    'skill_long_passing', 'power_shot_power', 'physic', 'skill_ball_control', 'shooting',
    'skill_curve', 'power_long_shots', 'mentality_aggression', 'attacking_crossing',
    'skill_fk_accuracy', 'attacking_volleys', 'skill_dribbling', 'power_stamina',
    'power_strength', 'mentality_positioning', 'attacking_heading_accuracy', 'mentality_penalties',
    'skill_moves', 'attacking_finishing', 'mentality_interceptions', 'defending',
    'power_jumping', 'defending_marking_awareness', 'movement_agility',
    'defending_standing_tackle', 'defending_sliding_tackle', 'weak_foot',
    'movement_sprint_speed', 'league_level', 'player_id'
]

def main():
    st.title("Football Players Rating Prediction")

    # Input data
    data = {}
    for col in expected_columns:
        data[col] = st.number_input(col, value=0)

    if st.button("Predict"):
        df = pd.DataFrame([data], columns=expected_columns)
        prediction = model.predict(df)
        st.write(f"Predicted Overall Rating: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()

