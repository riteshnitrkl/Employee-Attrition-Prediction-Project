# Streamlit Employee Attrition App

## Expected project structure

    Hexaware_OA/
    ├── app.py
    ├── requirements.txt
    ├── data/
    │   └── employee_attrition.csv
    ├── saved_models/
    │   └── top_8_LogisticRegression_Tuned.joblib
    └── outputs/

The app expects the tuned model bundle created by Step 20 to contain `pipeline` and `features`.

## Run on macOS

Open a terminal in `Hexaware_OA`, then run:

    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python -m streamlit run app.py

If a browser does not open automatically, use the local URL printed in the terminal (usually http://localhost:8501).

Keep `app.py` in the project root beside `data/` and `saved_models/`. Run Step 20 successfully first so the tuned model file exists.

The threshold slider changes the displayed class label, not the predicted probability. Validate any production threshold before use.
