import streamlit as st
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

st.set_page_config(page_title="GoalOracle ⚽", layout="wide")

# --- Helper functions ------------------------------------------------------

def calculate_score_probabilities(lambda_a, lambda_b, max_goals=8):
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i, j] = poisson.pmf(i, lambda_a) * poisson.pmf(j, lambda_b)
    return matrix

def calculate_outcome_probabilities(prob_matrix):
    win_a = np.tril(prob_matrix, -1).sum()
    draw = np.trace(prob_matrix)
    win_b = np.triu(prob_matrix, 1).sum()
    return win_a, draw, win_b

def most_probable_score(prob_matrix):
    idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
    return idx, prob_matrix[idx]

# --- Custom CSS ------------------------------------------------------------

st.markdown("""
<style>
.centered-header {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.input-header {
    text-align: center;
    color: #003366; 
    margin-top: -10px;
    margin-bottom: -5px;
}
.center-buttons {
    display: flex;
    justify-content: center;
    gap: 100px;
    margin-top: 30px;
    margin-bottom: 20px;
}
button[kind="secondary"], button[kind="primary"] {
    width: 600px !important;
    height: 60px !important;
    font-size: 20px !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}
button:hover {
    box-shadow: 0px 0px 20px rgba(0,208,192, 0.7) !important;
    background-color: #00D0C0 !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# --- Header Section (Visible Logo + Inline Title) --------------------------

import streamlit as st
from PIL import Image
import base64
from io import BytesIO

st.set_page_config(page_title="GoalOracle", layout="wide")

# Load image
logo = Image.open("Guluguluoracleaura.png")

# Resize proportionally
width, height = logo.size
scale_factor = 0.8
new_width = int(width * scale_factor)
new_height = int(height * scale_factor)
logo = logo.resize((new_width, new_height))

# Convert to base64
buffered = BytesIO()
logo.save(buffered, format="PNG")
encoded_logo = base64.b64encode(buffered.getvalue()).decode()

# Display centered with proper size
st.markdown(
    f"""
    <div style="display: flex; justify-content: center; align-items: center; margin-top: -20px; margin-bottom: -10px;">
        <img src="data:image/png;base64,{encoded_logo}" width="{new_width}" height="{new_height}">
    </div>
    """,
    unsafe_allow_html=True
)

# --- Layout Columns --------------------------------------------------------

col1, col3 = st.columns(2)

with col1:
    st.markdown("<h3 class='input-header'>Team A — Inputs</h3>", unsafe_allow_html=True)
    ta_goals = st.number_input("Goals Scored (λ)", min_value=0.0, value=1.2, step=0.1, format="%.2f", key="ta_goals")
    ta_conceded = st.number_input("Goals Conceded", min_value=0.0, value=1.0, step=0.1)
    ta_sot = st.number_input("Shots on Target", min_value=0.0, value=3.0, step=0.1)
    ta_chances = st.number_input("Chances Created", min_value=0.0, value=5.0, step=0.1)
    ta_poss = st.number_input("Possession (%)", min_value=0.0, max_value=100.0, value=52.0, step=0.1)
    ta_pass = st.number_input("Pass Completion (%)", min_value=0.0, max_value=100.0, value=82.0, step=0.1)

with col3:
    st.markdown("<h3 class='input-header'>Team B — Inputs</h3>", unsafe_allow_html=True)
    tb_goals = st.number_input("Goals Scored (λ)", min_value=0.0, value=1.0, step=0.1, format="%.2f", key="tb_goals")
    tb_conceded = st.number_input("Goals Conceded", min_value=0.0, value=1.1, step=0.1)
    tb_sot = st.number_input("Shots on Target", min_value=0.0, value=2.7, step=0.1)
    tb_chances = st.number_input("Chances Created", min_value=0.0, value=4.0, step=0.1)
    tb_poss = st.number_input("Possession (%)", min_value=0.0, max_value=100.0, value=48.0, step=0.1)
    tb_pass = st.number_input("Pass Completion (%)", min_value=0.0, max_value=100.0, value=79.0, step=0.1)

st.markdown("<div class='center-buttons'>", unsafe_allow_html=True)
predict = st.button("Predict")
reset = st.button("Reset")
st.markdown("</div>", unsafe_allow_html=True)

# --- Logic -----------------------------------------------------------------

if reset:
    for k in ["ta_goals", "tb_goals"]:
        if k in st.session_state:
            st.session_state[k] = 0.0
    st.experimental_rerun()

if predict:
    try:
        lambda_a = float(ta_goals)
        lambda_b = float(tb_goals)
        if lambda_a < 0 or lambda_b < 0:
            raise ValueError("Lambdas must be non-negative")

        prob_matrix = calculate_score_probabilities(lambda_a, lambda_b, max_goals=8)
        win_a, draw, win_b = calculate_outcome_probabilities(prob_matrix)
        (best_i, best_j), best_p = most_probable_score(prob_matrix)

        st.subheader("Prediction Results")
        st.write(f"Most Probable Score: {best_i} - {best_j} ({best_p:.2%})")
        st.write(f"Team A Win: {win_a:.2%}   |   Draw: {draw:.2%}   |   Team B Win: {win_b:.2%}")
        st.markdown("---")

        fig, ax = plt.subplots()
        im = ax.imshow(prob_matrix, origin='lower', aspect='auto')
        ax.set_xlabel('Team B Goals')
        ax.set_ylabel('Team A Goals')
        ax.set_title('Score Probability Matrix')
        for i in range(prob_matrix.shape[0]):
            for j in range(prob_matrix.shape[1]):
                p = prob_matrix[i, j]
                if p > 0.001:
                    ax.text(j, i, f"{p:.1%}", ha='center', va='center', fontsize=8)
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Invalid input detected: {e}")

st.markdown("---")
st.caption("GoalOracle — Poisson-based score prediction using the 'Goals Scored' inputs as λ for each team.")
st.markdown("[Visit GoalOracle GitHub](https://github.com/Rithukrish26/GoalOracle---Streamlit/tree/main)")
st.markdown("[GoalOracle for Mobile Phones](https://goaloracle---mobile.streamlit.app)")



















