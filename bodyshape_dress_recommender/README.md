# Body Shape & Dress Style Recommender

This project uses a decision tree model trained on synthetic body measurements to classify body shape and suggest dressing styles using rule-based logic.

### Features
- Predict body shape (Pear, Apple, Rectangle, Hourglass, Inverted Triangle)
- Suggest dress styles based on rules mapped to body shape
- Interactive user interface via Streamlit

---

### Project Structure

```
bodyshape_dress_recommender/
├── app/                    # Streamlit interface
│   └── streamlit_app.py
├── data/                   # Synthetic training data
│   └── synthetic_body_data.csv
├── model/                  # Saved decision tree model
│   └── body_shape_tree.pkl
├── src/                    # Core logic
│   ├── data_generator.py
│   ├── predictor.py
│   ├── style_mapping.py
│   ├── train_model.py
│   └── **init**.py
├── test_predictor.py       # Script to test prediction manually
├── requirements.txt
└── README.md
````
---

### How to Run

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

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
