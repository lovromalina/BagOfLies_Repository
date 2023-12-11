from sklearn.datasets import fetch_olivetti_faces

data = fetch_olivetti_faces()
X, y = data.data, data.target

print(X)

