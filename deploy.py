import spacy

model_path = 'output_model'  # Update with the path to your saved model
nlp_cm = spacy.load(model_path)

# Process test text
text = """
DOB: 17/12/1994
Subjective:
Patient presents today for a follow-up visit regarding chronic hypertension. Reports occasional headaches but denies any chest pain, shortness of breath, or dizziness. Compliant with prescribed medications.

Objective:

Vitals: BP readings consistently elevated around 120/90 mmHg during office visits.
General: Patient appears well-nourished and in no acute distress.
CV (Cardiovascular): Regular rate and rhythm, no murmurs.
Neurological: No focal deficits noted.
"""
doc_cm = nlp_cm(text)
for ent in doc_cm.ents:
    print("{} - {}".format(ent.text, ent.label_))

