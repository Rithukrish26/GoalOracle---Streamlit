import streamlit as st
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from PIL import Image
import base64
from io import BytesIO

# --- Page Config ---
st.set_page_config(page_title="GoalOracle Mobile ⚽", layout="centered")

# --- Defaults ---
DEFAULTS = {
    "ta_goals": 1.2,
    "ta_conceded": 1.0,
    "ta_sot": 3.0,
    "ta_chances": 5.0,
    "ta_poss": 52.0,
    "ta_pass": 82.0,
    "tb_goals": 1.0,
    "tb_conceded": 1.1,
    "tb_sot": 2.7,
    "tb_chances": 4.0,
    "tb_poss": 48.0,
    "tb_pass": 79.0,
}

# --- Helper Functions ---
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

# --- Reset callback (no experimental_rerun) ---
def reset_inputs():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

# --- Custom CSS for hover-glow buttons ---
st.markdown("""
<style>
/* Default black buttons */
.stButton>button, div.stNumberInput > div > button {
    background-color: black !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: bold;
    height: 48px !important;
    width: 120px !important;
    border-radius: 12px !important;
    border: none !important;
    transition: all 0.25s ease;
}

/* Glow and cyan background on hover (like your screenshot) */
.stButton>button:hover, div.stNumberInput > div > button:hover {
    background-color: #00D0C0 !important;
    color: black !important;
    box-shadow: 0 0 10px #00D0C0, 0 0 20px #00D0C0, 0 0 30px #00D0C0;
}

/* Make the main buttons wider on small screens */
@media (max-width: 600px) {
  .stButton>button {
    width: 90% !important;
  }
}

/* small spacing tweaks */
.css-1kyxreq {padding-top: 4px;} /* safe minor tweak for top padding on some themes */
</style>
""", unsafe_allow_html=True)

# --- Logo Section ---
try:
    logo = Image.open("Guluguluoracleaura.png")
    width, height = logo.size
    scale_factor = 0.5
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    logo = logo.resize((new_width, new_height))

    buffered = BytesIO()
    logo.save(buffered, format="PNG")
    encoded_logo = base64.b64encode(buffered.getvalue()).decode()

    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center; margin-top: 0px; margin-bottom: 10px;">
            <img src="data:image/png;base64,{encoded_logo}" width="{new_width}" height="{new_height}">
        </div>
        """,
        unsafe_allow_html=True
    )
except Exception:
    # If logo missing, don't crash app
    st.markdown("<h2 style='text-align:center'>GoalOracle Mobile ⚽</h2>", unsafe_allow_html=True)

# --- Inputs (use keys so reset can target them) ---
st.markdown("<h3 style='text-align:center; color:#003366; margin-top:5px; margin-bottom:5px;'>Team A — Inputs</h3>", unsafe_allow_html=True)
ta_goals = st.number_input("Goals Scored (λ)", min_value=0.0, value=DEFAULTS["ta_goals"], step=0.1, format="%.2f", key="ta_goals")
ta_conceded = st.number_input("Goals Conceded", min_value=0.0, value=DEFAULTS["ta_conceded"], step=0.1, key="ta_conceded")
ta_sot = st.number_input("Shots on Target", min_value=0.0, value=DEFAULTS["ta_sot"], step=0.1, key="ta_sot")
ta_chances = st.number_input("Chances Created", min_value=0.0, value=DEFAULTS["ta_chances"], step=0.1, key="ta_chances")
ta_poss = st.number_input("Possession (%)", min_value=0.0, max_value=100.0, value=DEFAULTS["ta_poss"], step=0.1, key="ta_poss")
ta_pass = st.number_input("Pass Completion (%)", min_value=0.0, max_value=100.0, value=DEFAULTS["ta_pass"], step=0.1, key="ta_pass")

st.markdown("<h3 style='text-align:center; color:#003366; margin-top:10px; margin-bottom:5px;'>Team B — Inputs</h3>", unsafe_allow_html=True)
tb_goals = st.number_input("Goals Scored (λ)", min_value=0.0, value=DEFAULTS["tb_goals"], step=0.1, format="%.2f", key="tb_goals")
tb_conceded = st.number_input("Goals Conceded", min_value=0.0, value=DEFAULTS["tb_conceded"], step=0.1, key="tb_conceded")
tb_sot = st.number_input("Shots on Target", min_value=0.0, value=DEFAULTS["tb_sot"], step=0.1, key="tb_sot")
tb_chances = st.number_input("Chances Created", min_value=0.0, value=DEFAULTS["tb_chances"], step=0.1, key="tb_chances")
tb_poss = st.number_input("Possession (%)", min_value=0.0, max_value=100.0, value=DEFAULTS["tb_poss"], step=0.1, key="tb_poss")
tb_pass = st.number_input("Pass Completion (%)", min_value=0.0, max_value=100.0, value=DEFAULTS["tb_pass"], step=0.1, key="tb_pass")

# --- Buttons: use on_click for reset to avoid experimental_rerun issues ---
col1, col2 = st.columns([1,1])
with col1:
    predict = st.button("Predict")
with col2:
    reset = st.button("Reset", on_click=reset_inputs)

# --- Logic ---
if predict:
    try:
        lambda_a = float(st.session_state.get("ta_goals", DEFAULTS["ta_goals"]))
        lambda_b = float(st.session_state.get("tb_goals", DEFAULTS["tb_goals"]))
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
        im = ax.imshow(prob_matrix, origin='lower', aspect='auto', cmap='coolwarm')
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
st.caption("GoalOracle — Mobile Poisson-based score prediction using the 'Goals Scored' inputs as λ for each team.") 
