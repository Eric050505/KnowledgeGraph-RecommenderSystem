from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import util

X = util.load_data('classification_train_data.pkl')
y = util.load_data('classification_train_label.pkl')
X = X[:, 1:]
y = y[:, 1:].reshape(-1)
scaler = StandardScaler()
scaler.fit(X)
util.save_data('classification_scaler.pkl', scaler)

X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm_clf = SVC(kernel='rbf', gamma='scale', C=1.0, verbose=True, max_iter=1000)
svm_clf.fit(X_train, y_train)
util.save_data('classification_svm_model.pkl', svm_clf)
y_pred = svm_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
