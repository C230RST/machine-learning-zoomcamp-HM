import pickle

with open("pipeline.bin", "rb") as f:
    model = pickle.load(f)
lead = {
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}
probabilities = model.predict_proba([lead])[0]

classes = list(model.classes_)
converted_index = classes.index(1) if 1 in classes else classes.index("converted")

probability = probabilities[converted_index]

print(f"Conversion probability: {probability:.3f}")