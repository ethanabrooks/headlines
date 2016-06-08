import os

for filename in os.listdir('.'):
    if filename.endswith('.pkl'):
        with open(filename) as handle:
            tensor =
