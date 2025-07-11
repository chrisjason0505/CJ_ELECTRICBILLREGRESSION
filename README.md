# IGNITE-HACKATHON-AI-ML-
# ⚡ Predicting Electricity Bills with Machine Learning

Welcome to the **Electricity Bill Predictor** — a simple and practical machine learning project that estimates your monthly electricity bill based on how much you use your home appliances.

✅ **Live App Demo:** [Enjoy!!](https://ibwburtv8qn8rheqfv2yqn.streamlit.app/) 

---

## 💡 What Does This Project Do?

This project helps users:

- Predict their monthly electricity bill before it arrives
- Understand which appliances consume the most electricity
- Plan and save on energy costs

It uses:

- Appliance usage data (in kWh)
- Local tariff rates

---

## 🚀 Streamlit Web App

We’ve deployed this project as a **Streamlit web app** for easy use in the browser — no coding required.

**How it works:**

1. Visit the web app
2. Enter how much electricity your appliances use
3. Select your monthly hour usage and  tariff rate
4. Click **Predict**
5. Instantly see your estimated electricity bill!

✅ Try the app here (dummy URL):

[https://my-electricity-predictor.streamlit.app](https://my-electricity-predictor.streamlit.app)

---

## 📁 Why We Use `.pkl` Files

Once our machine learning model is trained, we don’t want to retrain it every time the app runs. That’s where `.pkl` (pickle) files come in.

A `.pkl` file lets us:

- Save the trained model once
- Load it quickly inside our Streamlit app
- Make instant predictions without extra training

**Think of it like saving game progress:**  
The model has already learned the patterns, and we’re simply loading that knowledge to make predictions fast.

Benefits of using `.pkl`:

✅ Fast loading  
✅ No repeated training  
✅ Perfect for deploying models in web apps

---

## 📊 Model Performance

Our model performs very well on test data:

- **Mean Squared Error (MSE):** 4969.58  
    → Predictions are quite close to real bills.

- **R-squared Score (R²):** 0.9956  
    → The model explains ~99.56% of variations in electricity bills.

- **Feature Importance:**
    - Monthly Appliance Usage Hours → strongest impact
    - Tariff Rate → heavily influences the bill
    - Smaller impact from individual appliances

In short: **our model is accurate and practical for real-life predictions.**

---

## 🛠 How to Run Locally

Want to run this on your own computer?

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/electricity-bill-predictor.git
    cd electricity-bill-predictor
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

Make sure your `.pkl` file (saved trained model) is in the project folder so the app can load it.

---

## ✅ Future Improvements

- Add more appliances
- Support tariffs for other countries
- Improve the user interface
- Allow saving prediction history

---

Thanks for checking out this project!

**Built with ❤️ Python + Streamlit**

