from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import random
import joblib

RANDOM_SEED = 42
N_JOBS = -1

def tokenize(data):
    return [
        (
            word_tokenize(sample[0]),
            sample[1]
        )
        for sample in data
    ]

def remove_stopwords(data):
    stop_words = set(stopwords.words('english'))
    return [
        (
            [word for word in sample[0] if word not in stop_words],
            sample[1]
        )
        for sample in data
    ]

def train_test_split(fake_data, true_data, split=0.1):
    fake_data = [(' '.join(sample[0]), sample[1]) for sample in fake_data]
    true_data = [(' '.join(sample[0]), sample[1]) for sample in true_data]

    random.seed(RANDOM_SEED)

    fake_test = random.sample(fake_data, int(len(fake_data) * split))
    fake_train = [sample for sample in fake_data if not fake_test.__contains__(sample)]
    true_test = random.sample(true_data, int(len(true_data) * split))
    true_train = [sample for sample in true_data if not true_test.__contains__(sample)]

    train_data = fake_train + true_train
    test_data = fake_test + true_test

    random.shuffle(train_data)
    random.shuffle(test_data)

    X_train = [sample[0] for sample in train_data]
    y_train = [sample[1] for sample in train_data]
    X_test = [sample[0] for sample in test_data]
    y_test = [sample[1] for sample in test_data]
    return X_train, y_train, X_test, y_test

fake_df = pd.read_csv('MathProject/data/Fake.csv')
true_df = pd.read_csv('MathProject/data/True.csv')

fake_data = [(f'{fake_df.iloc[index]["title"]}. {fake_df.iloc[index]["text"]}', 1) for index in range(fake_df.shape[0])]

print(len(fake_data))

true_data = []
for index in range(true_df.shape[0]):
    title = true_df.iloc[index]['title']
    text = true_df.iloc[index]['text']
    if text.__contains__('(Reuters) - '):
        text = text[text.index('-') + 2:]
    true_data.append((f'{title}. {text}', 0))

print(len(true_data))

data = true_data + fake_data
print(len(data))

true_data = tokenize(true_data)
fake_data = tokenize(fake_data)

true_data = remove_stopwords(true_data)
fake_data = remove_stopwords(fake_data)

X_train, y_train, X_test, y_test = train_test_split(fake_data, true_data)

feature_extractor = TfidfVectorizer(use_idf=True,
                                    sublinear_tf=False,
                                    strip_accents='ascii',
                                    smooth_idf=False,
                                    norm='l1',
                                    ngram_range=(2, 2), 
                                    lowercase=False, 
                                    decode_error='replace', 
                                    binary=False, 
                                    analyzer='char_wb')

classifier = RandomForestClassifier(warm_start=True,
                                    random_state=42,
                                    n_jobs=-1,
                                    n_estimators=1000,
                                    min_samples_split=7,
                                    min_samples_leaf=4,
                                    max_features='auto',
                                    max_depth=70.0,
                                    class_weight='balanced',
                                    bootstrap=False)

pipeline = Pipeline([
    ('feature_extractor', feature_extractor),
    ('classifier', classifier),
])

pipeline.fit(X_train, y_train)
print(X_test)
print(y_test)

y_pred = pipeline.predict(X_test)
print(accuracy_score(y_test, y_pred))

joblib.dump(pipeline, 'MathProject/model.joblib')