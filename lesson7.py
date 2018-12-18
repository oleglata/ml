import numpy as np
import nltk
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)

twenty_train.data[0]

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

_ = text_clf.fit(twenty_train.data, twenty_train.target)

predicted = text_clf.predict(twenty_test.data)
print(np.mean(predicted == twenty_test.target))

text_lr = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('lr', LogisticRegression(multi_class='multinomial',solver ='newton-cg'))])
_ = text_lr.fit(twenty_train.data, twenty_train.target)

predicted = text_lr.predict(twenty_test.data)
print(np.mean(predicted == twenty_test.target))


text_lr = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()),
                    ('lr', LogisticRegression(multi_class='multinomial',solver ='newton-cg'))])
_ = text_lr.fit(twenty_train.data, twenty_train.target)

predicted = text_lr.predict(twenty_test.data)
print(np.mean(predicted == twenty_test.target))


text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])
_ = text_clf_svm.fit(twenty_train.data, twenty_train.target)

predicted = text_clf_svm.predict(twenty_test.data)
print(np.mean(predicted == twenty_test.target))

text_rf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                    ('rf', RandomForestClassifier(n_estimators=10, criterion='entropy'))])

_ = text_rf.fit(twenty_train.data, twenty_train.target)

predicted = text_rf.predict(twenty_test.data)
print(np.mean(predicted == twenty_test.target))


parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf-svm__alpha': (1e-2, 1e-3)}
gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(twenty_train.data, twenty_train.target)


gs_clf_svm.best_score_
gs_clf_svm.best_params_
parameters_nb = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False)}
gs_clf = GridSearchCV(text_clf, parameters_nb, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
gs_clf.best_score_
gs_clf.best_params_

stemmer = SnowballStemmer("english")

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
                             ('mnb', MultinomialNB(fit_prior=False))])

text_mnb_stemmed = text_mnb_stemmed.fit(twenty_train.data, twenty_train.target)
predicted_mnb_stemmed = text_mnb_stemmed.predict(twenty_test.data)
print(np.mean(predicted_mnb_stemmed == twenty_test.target))


stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

text_clf_svm_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])

_ = text_clf_svm_stemmed.fit(twenty_train.data, twenty_train.target)

predicted = text_clf_svm_stemmed.predict(twenty_test.data)
print(np.mean(predicted == twenty_test.target))


stemmed_count_vect = StemmedCountVectorizer(stop_words='english', ngram_range=(1,2))

text_clf_svm_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])

_ = text_clf_svm_stemmed.fit(twenty_train.data, twenty_train.target)

predicted = text_clf_svm_stemmed.predict(twenty_test.data)
print(np.mean(predicted == twenty_test.target))