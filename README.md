# Sentiment Analysis Project
This project contains two sentiment analysis programs for Hotel Reviews using a Hotel Reviews dataset from Datafiniti. The training models for this Machine Learning project are built through Count Vectorizer (for the countvectorizer.py program) and TF-IDF Vectorizer (for the tdidf.py program). You can see the difference in implementation and accuracy results through both types of Vectorizers by running the programs separately (usually, TF-IDF Vectorizer is considered more accurate).

## System  Requirements
Use the pip install command to install the following imports:
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
```

## Usage (description of actions performed)
```
1. dataset imported
2. null values deleted
3. 30% representative sample is taken to avoid slow down of system
4. sentiments column added
5. input training features and labels defined
6. dataset split into training sets and testing sets
7. text data vectorizer (using CountVectorizer or TF-IDF Vectorizer)
8. models trained:
 -  Logistic Regression (linear clasification)
 -  Support Vector Machine (linear/non-linear data separated into classes by a line/hyperplane)
 -  K Nearest Neighbor (local approximation)
9. print Accuracy Scores, Confusion Matrix, Ture Positive and Negative Rates for all three models

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
MIT
