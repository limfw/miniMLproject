
```markdown
# Body Shape & Dress Style Recommender

This project uses a decision tree model trained on synthetic body measurements to classify body shape and suggest dressing styles using rule-based logic.

### Features
- Predict body shape (Pear, Apple, Rectangle, Hourglass, Inverted Triangle)
- Suggest dress styles based on rules mapped to body shape
- Interactive user interface via Streamlit

---

### Project Structure

```

bodyshape\_dress\_recommender/
├── app/                    # Streamlit interface
│   └── streamlit\_app.py
├── data/                   # Synthetic training data
│   └── synthetic\_body\_data.csv
├── model/                  # Saved decision tree model
│   └── body\_shape\_tree.pkl
├── src/                    # Core logic
│   ├── data\_generator.py
│   ├── predictor.py
│   ├── style\_mapping.py
│   ├── train\_model.py
│   └── **init**.py
├── test\_predictor.py       # Script to test prediction manually
├── requirements.txt
└── README.md

````

---

### How to Run

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
````

2. **Generate synthetic data**

   ```bash
   python src/data_generator.py
   ```

3. **Train the model**

   ```bash
   python src/train_model.py
   ```

4. **Test the predictor**

   ```bash
   python test_predictor.py
   ```

5. **Launch the Streamlit App**

   ```bash
   streamlit run app/streamlit_app.py
   ```

---

### Example Input

| Shoulders | Bust | Waist | Hips |
| --------- | ---- | ----- | ---- |
| 39        | 36   | 29    | 40   |

---

###  Future Enhancements

* Add image recommendations (via `assets/`)
* Allow user feedback for learning
* Support style suggestions for different occasions

---

### License

This project is for educational and demonstration purposes.

---

```
