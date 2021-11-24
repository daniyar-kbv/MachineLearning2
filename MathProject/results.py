from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

results = [
    {
        'best_extractor': HashingVectorizer(analyzer='char',
                                            binary=True,
                                            lowercase=False,
                                            ngram_range=(2, 2),
                                            strip_accents='ascii'),
        'model': MultinomialNB(alpha=0.0, fit_prior=False),
        'parameters': {
            'feature_extractor__strip_accents': 'ascii',
            'feature_extractor__ngram_range': (2, 2),
            'feature_extractor__lowercase': False,
            'feature_extractor__decode_error': 'strict',
            'feature_extractor__binary': True,
            'feature_extractor__analyzer': 'char',
            'classifier__fit_prior': False,
            'classifier__alpha': 0.0
        },
        'score': 0.9828469592336824
    },
    {
        'best_extractor': HashingVectorizer(binary=True,
                                            decode_error='ignore',
                                            lowercase=False,
                                            strip_accents='ascii'),
        'model': LogisticRegression(class_weight='balanced',
                                    fit_intercept=False,
                                    multi_class='ovr',
                                    n_jobs=-1,
                                    random_state=42,
                                    solver='liblinear'),
        'parameters': {
            'feature_extractor__strip_accents': 'ascii',
            'feature_extractor__ngram_range': (1, 1),
            'feature_extractor__lowercase': False,
            'feature_extractor__decode_error': 'ignore',
            'feature_extractor__binary': True,
            'feature_extractor__analyzer': 'word',
            'classifier__warm_start': False,
            'classifier__solver': 'liblinear',
            'classifier__random_state': 42,
            'classifier__n_jobs': -1,
            'classifier__multi_class': 'ovr',
            'classifier__fit_intercept': False,
            'classifier__class_weight': 'balanced',
            'classifier__C': 1.0
        },
        'score': 0.9924259300512364
    },
    {
        'best_extractor': HashingVectorizer(analyzer='char',
                                            ngram_range=(2, 2),
                                            strip_accents='unicode'),
        'model': RandomForestClassifier(bootstrap=False,
                                        class_weight='balanced_subsample',
                                        max_depth=200.0,
                                        min_samples_split=4,
                                        n_estimators=155,
                                        n_jobs=-1,
                                        random_state=42,
                                        warm_start=True),
        'parameters': {
            'feature_extractor__strip_accents': 'unicode',
            'feature_extractor__ngram_range': (2, 2),
            'feature_extractor__lowercase': True,
            'feature_extractor__decode_error': 'strict',
            'feature_extractor__binary': False,
            'feature_extractor__analyzer': 'char',
            'classifier__warm_start': True,
            'classifier__random_state': 42,
            'classifier__n_jobs': -1,
            'classifier__n_estimators': 155,
            'classifier__min_samples_split': 4,
            'classifier__min_samples_leaf': 1,
            'classifier__max_features': 'auto',
            'classifier__max_depth': 200.0,
            'classifier__class_weight': 'balanced_subsample',
            'classifier__bootstrap': False
        },
        'score': 0.9759411895745155
    },
    {
        'best_extractor': HashingVectorizer(analyzer='char_wb',
                                            binary=True,
                                            decode_error='replace',
                                            ngram_range=(2, 2),
                                            strip_accents='unicode'),
        'model': KNeighborsClassifier(algorithm='ball_tree',
                                      n_jobs=-1,
                                      n_neighbors=2,
                                      weights='distance'),
        'parameters': {
            'feature_extractor__strip_accents': 'unicode',
            'feature_extractor__ngram_range': (2, 2),
            'feature_extractor__lowercase': True,
            'feature_extractor__decode_error': 'replace',
            'feature_extractor__binary': True,
            'feature_extractor__analyzer': 'char_wb',
            'classifier__weights': 'distance',
            'classifier__p': 2,
            'classifier__n_neighbors': 2,
            'classifier__n_jobs': -1,
            'classifier__algorithm': 'ball_tree'
        },
        'score': 0.9356204054355091},
    {
        'best_extractor': CountVectorizer(analyzer='char_wb',
                                          binary=True,
                                          decode_error='ignore',
                                          lowercase=False,
                                          ngram_range=(2, 2),
                                          strip_accents='unicode'),
        'model': MultinomialNB(alpha=0.3333333333333333),
        'parameters': {
            'classifier__alpha': 0.3333333333333333,
            'classifier__fit_prior': True,
            'feature_extractor__analyzer': 'char_wb',
            'feature_extractor__binary': True,
            'feature_extractor__decode_error': 'ignore',
            'feature_extractor__lowercase': False,
            'feature_extractor__ngram_range': (2, 2),
            'feature_extractor__strip_accents': 'unicode'
        },
        'score': 0.9855201603920695
    },
    {
        'best_extractor': CountVectorizer(binary=True,
                                          decode_error='ignore',
                                          lowercase=False,
                                          strip_accents='ascii'),
        'model': LogisticRegression(class_weight='balanced',
                                    fit_intercept=False,
                                    multi_class='ovr',
                                    n_jobs=-1,
                                    random_state=42,
                                    solver='liblinear'),
        'parameters': {
            'classifier__C': 1.0,
            'classifier__class_weight': 'balanced',
            'classifier__fit_intercept': False,
            'classifier__multi_class': 'ovr',
            'classifier__n_jobs': -1,
            'classifier__random_state': 42,
            'classifier__solver': 'liblinear',
            'classifier__warm_start': False,
            'feature_extractor__analyzer': 'word',
            'feature_extractor__binary': True,
            'feature_extractor__decode_error': 'ignore',
            'feature_extractor__lowercase': False,
            'feature_extractor__ngram_range': (1, 1),
            'feature_extractor__strip_accents': 'ascii'
        },
        'score': 0.9955446647360214
    },
    {
        'best_extractor': CountVectorizer(analyzer='char',
                                          ngram_range=(2, 2),
                                          strip_accents='unicode'),
        'model': RandomForestClassifier(bootstrap=False,
                                        class_weight='balanced_subsample',
                                        max_depth=200.0,
                                        min_samples_split=4,
                                        n_estimators=155,
                                        n_jobs=-1,
                                        random_state=42,
                                        warm_start=True),
        'parameters': {
            'classifier__bootstrap': False,
            'classifier__class_weight': 'balanced_subsample',
            'classifier__max_depth': 200.0,
            'classifier__max_features': 'auto',
            'classifier__min_samples_leaf': 1,
            'classifier__min_samples_split': 4,
            'classifier__n_estimators': 155,
            'classifier__n_jobs': -1,
            'classifier__random_state': 42,
            'classifier__warm_start': True,
            'feature_extractor__analyzer': 'char',
            'feature_extractor__binary': False,
            'feature_extractor__decode_error': 'strict',
            'feature_extractor__lowercase': True,
            'feature_extractor__ngram_range': (2, 2),
            'feature_extractor__strip_accents': 'unicode'
        },
        'score': 0.9946535976832257
    },
    {
        'best_extractor': CountVectorizer(analyzer='char_wb',
                                          binary=True,
                                          decode_error='replace',
                                          ngram_range=(2, 2),
                                          strip_accents='unicode'),
        'model': KNeighborsClassifier(algorithm='ball_tree',
                                      n_jobs=-1,
                                      n_neighbors=2,
                                      weights='distance'),
        'parameters': {
            'feature_extractor__strip_accents': 'unicode',
            'feature_extractor__ngram_range': (2, 2),
            'feature_extractor__lowercase': True,
            'feature_extractor__decode_error': 'replace',
            'feature_extractor__binary': True,
            'feature_extractor__analyzer': 'char_wb',
            'classifier__weights': 'distance',
            'classifier__p': 2,
            'classifier__n_neighbors': 2,
            'classifier__n_jobs': -1,
            'classifier__algorithm': 'ball_tree'
        },
        'score': 0.9240365337491646
    },
    {
        'best_extractor': TfidfVectorizer(analyzer='char', binary=True, decode_error='replace',
                                          lowercase=False, ngram_range=(1, 2), strip_accents='unicode',
                                          sublinear_tf=True, use_idf=False),
        'model': MultinomialNB(alpha=0.0),
        'parameters': {
            'feature_extractor__use_idf': False, 'feature_extractor__sublinear_tf': True,
            'feature_extractor__strip_accents': 'unicode', 'feature_extractor__smooth_idf': True,
            'feature_extractor__norm': 'l2', 'feature_extractor__ngram_range': (1, 2),
            'feature_extractor__lowercase': False, 'feature_extractor__decode_error': 'replace',
            'feature_extractor__binary': True, 'feature_extractor__analyzer': 'char', 'classifier__fit_prior': True,
            'classifier__alpha': 0.0}, 'score': 0.9848518601024727
    },
    {'best_extractor': TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 2), strip_accents='unicode',
                                       sublinear_tf=True),
     'model': LogisticRegression(C=0.4, fit_intercept=False, n_jobs=-1, random_state=42,
                                 solver='newton-cg'),
     'parameters': {'feature_extractor__use_idf': True, 'feature_extractor__sublinear_tf': True,
                    'feature_extractor__strip_accents': 'unicode', 'feature_extractor__smooth_idf': True,
                    'feature_extractor__norm': 'l2', 'feature_extractor__ngram_range': (2, 2),
                    'feature_extractor__lowercase': True, 'feature_extractor__decode_error': 'strict',
                    'feature_extractor__binary': False, 'feature_extractor__analyzer': 'char_wb',
                    'classifier__warm_start': False, 'classifier__solver': 'newton-cg', 'classifier__random_state': 42,
                    'classifier__n_jobs': -1, 'classifier__multi_class': 'auto', 'classifier__fit_intercept': False,
                    'classifier__class_weight': None, 'classifier__C': 0.4}, 'score': 0.993539763867231},
    {'best_extractor': TfidfVectorizer(analyzer='char_wb', decode_error='replace', lowercase=False,
                                       ngram_range=(2, 2), norm='l1', smooth_idf=False,
                                       strip_accents='ascii'),
     'model': RandomForestClassifier(bootstrap=False, class_weight='balanced', max_depth=70.0,
                                     min_samples_leaf=4, min_samples_split=7,
                                     n_estimators=1000, n_jobs=-1, random_state=42,
                                     warm_start=True),
     'parameters': {'feature_extractor__use_idf': True, 'feature_extractor__sublinear_tf': False,
                    'feature_extractor__strip_accents': 'ascii', 'feature_extractor__smooth_idf': False,
                    'feature_extractor__norm': 'l1', 'feature_extractor__ngram_range': (2, 2),
                    'feature_extractor__lowercase': False, 'feature_extractor__decode_error': 'replace',
                    'feature_extractor__binary': False, 'feature_extractor__analyzer': 'char_wb',
                    'classifier__warm_start': True, 'classifier__random_state': 42, 'classifier__n_jobs': -1,
                    'classifier__n_estimators': 1000, 'classifier__min_samples_split': 7,
                    'classifier__min_samples_leaf': 4, 'classifier__max_features': 'auto',
                    'classifier__max_depth': 70.0, 'classifier__class_weight': 'balanced',
                    'classifier__bootstrap': False}, 'score': 0.9959901982624193},
    {
        'best_extractor': TfidfVectorizer(analyzer='char', decode_error='replace', lowercase=False,
ngram_range=(1, 2), strip_accents='unicode', use_idf=False),
        'model': KNeighborsClassifier(algorithm='brute', n_jobs=-1, n_neighbors=2, p=1,
weights='distance'),
        'parameters': {'feature_extractor__use_idf': False, 'feature_extractor__sublinear_tf': False, 'feature_extractor__strip_accents': 'unicode', 'feature_extractor__smooth_idf': True, 'feature_extractor__norm': 'l2', 'feature_extractor__ngram_range': (1, 2), 'feature_extractor__lowercase': False, 'feature_extractor__decode_error': 'replace', 'feature_extractor__binary': False, 'feature_extractor__analyzer': 'char', 'classifier__weights': 'distance', 'classifier__p': 1, 'classifier__n_neighbors': 2, 'classifier__n_jobs': -1, 'classifier__algorithm': 'brute'},
        'score': 0.9173535308531967},
]

results_str = ''
for result in sorted(results, key=lambda x: x.get('score'), reverse=True):
    parameters_str = ''
    for param in list(result['parameters'].items()):
        parameters_str += f'\n  {param[0]}: {param[1]}'
    results_str += f"""Feature extractor: {result['best_extractor'].__class__.__name__}
Classifier: {result['model'].__class__.__name__}
Score: {result['score']}
Best parameters: {parameters_str}

"""

with open('MathProject/results.txt', 'w') as f:
    f.write(results_str)

