import re

with open('src/models/vision/mobilenet_tiny.c', 'r') as f:
    text = f.read()

# Since writing a full parser to refactor mobilenet_tiny is hard, I will do it systematically.
