# 🔥 Wildfire Evacuation Threat Predictor 🔥 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://wildfire-evac-app.streamlit.app/) 

## Overview
When a wildfire ignites, emergency managers face an impossible question with incomplete information: *Which fires will reach populated areas? How quickly? And which communities should prepare first?*

This web application predicts the probability that a wildfire will threaten an evacuation zone (within 5km) at **12, 24, 48, and 72-hour horizons**, drawing on telemetry data from just the **first five hours** after ignition. 

This project was built for the **WiDS Global Datathon 2026** and utilizes Right-Censored Survival Analysis to provide calibrated risk estimates for emergency responders.

Standard classification models fail at this task because they don't understand time-bound, censored data (e.g., a fire that hasn't hit an evacuation zone *yet*). This app utilizes **Survival Analysis** to generate continuous, monotonic survival curves.

The app's predictions are powered by an ensemble of two models:
1. **Random Survival Forest (RSF):** Highly robust at ranking the comparative urgency of different fires (optimizing the C-index).
2. **Gradient Boosting Survival Analysis (GBSA):** Excels at producing highly calibrated exact probabilities (optimizing the Brier Score).


Before predictions are made, the app engineers custom domain-specific features from the raw inputs:
* **Estimated Time of Arrival (ETA):** Remaining distance to the 5km threshold divided by closing speed.
* **Danger Index:** A composite metric weighing the fire's initial area, growth rate, and proximity to the evacuation zone.
* **Directed Threat Speed:** The raw speed of the fire mathematically penalized if the fire is not moving directly toward the evacuation zone.

## How to Run Locally

If you want to run this application on your own machine, follow these steps:

**1. Clone the repository**
```bash
git clone [https://github.com/](https://github.com/)<your-username>/wildfire-evac-app.git
cd wildfire-evac-app
```

**2. Install Dependencies**
Make sure you have Python 3 installed, then run:

```bash
pip install -r requirements.txt
```

**3. Run the Streamlit App**

```bash
streamlit run app.py
```

A browser window will automatically open to http://localhost:8501 hosting the app.

## Repository Structure

app.py: The main Streamlit application code.

requirements.txt: The Python packages required to run the app.

wildfire_rsf_model.pkl: The serialized, pre-trained Random Survival Forest model.

wildfire_gbsa_model.pkl: The serialized, pre-trained Gradient Boosting Survival model.

## Acknowledgments
Data and problem framing provided by the WiDS Global Datathon 2026 and WatchDuty.
