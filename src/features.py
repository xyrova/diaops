"""Single source of truth for the dataset schema.

Used by the validate stage, the prepare/train/evaluate stages and the API, so the
columns and allowed ranges can never drift apart between training and serving.
"""

TARGET = "Diabetes_binary"

# Allowed (min, max) for every feature, in the column order of the CSV.
# Binary flags are (0, 1). Ranges come from the BRFSS 2015 codebook.
FEATURE_RANGES = {
    "HighBP": (0, 1),
    "HighChol": (0, 1),
    "CholCheck": (0, 1),
    "BMI": (10, 100),  # sanity bounds, not a clinical range
    "Smoker": (0, 1),
    "Stroke": (0, 1),
    "HeartDiseaseorAttack": (0, 1),
    "PhysActivity": (0, 1),
    "Fruits": (0, 1),
    "Veggies": (0, 1),
    "HvyAlcoholConsump": (0, 1),
    "AnyHealthcare": (0, 1),
    "NoDocbcCost": (0, 1),
    "GenHlth": (1, 5),  # 1 = excellent ... 5 = poor
    "MentHlth": (0, 30),  # days in the past 30
    "PhysHlth": (0, 30),  # days in the past 30
    "DiffWalk": (0, 1),
    "Sex": (0, 1),
    "Age": (1, 13),  # 5-year age bands
    "Education": (1, 6),
    "Income": (1, 8),
}

FEATURES = list(FEATURE_RANGES)
COLUMNS = [TARGET] + FEATURES
