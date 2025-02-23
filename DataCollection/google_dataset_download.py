import json

import requests

# Load json file.

with open('ontology.json') as f:
    ontology = json.load(f)

# Get all classes in the ontology.
