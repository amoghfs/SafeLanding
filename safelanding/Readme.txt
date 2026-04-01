🚀 AUTONOMOUS MARS LANDING PROJECT

---

## 📁 PROJECT STRUCTURE

Auburn_1/        → Dataset (Mars terrain images)
app.py           → Streamlit app (UI + prediction)
main.py          → Training script
model.pth        → Trained AI model

---

## ⚙️ STEP 1: INSTALL REQUIRED LIBRARIES

Open terminal and run:

pip install torch torchvision opencv-python streamlit matplotlib numpy

---

## 🧠 STEP 2: TRAIN THE MODEL (ONLY ONCE)

Run this command:

python main.py

This will:

* Train the AI model
* Save the model as "model.pth"

---

## 🚀 STEP 3: RUN THE APP

Run this command:

streamlit run app.py

---

## 🖼️ STEP 4: USE THE APP

1. Upload a Mars image
2. AI will:

   * Break image into small squares
   * Detect terrain type
   * Create hazard map
3. It will show:

   * Safe landing zone
   * Safety score

---

## 🧑‍🏫 SIMPLE EXPLANATION

The AI looks at small parts of the image
and decides:

* Safe (flat ground)
* Risky (dunes)
* Dangerous (craters, rocks)

Then it finds the safest place to land 🚀

---

## ✅ DONE!

Enjoy exploring Mars with AI!
