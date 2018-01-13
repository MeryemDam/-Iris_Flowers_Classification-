
>>> import numpy as np
>>> import pylab as pl
>>> from mpl_toolkits.mplot3d import Axes3D
>>> from sklearn.cluster import Kmeans

>>> import sklearn.cluster
>>> from sklearn.cluster import KMeans
>>> 
>>> from sklearn import datasets
>>> np.random.seed(5)
>>> centers=[[1,1],[-1,-1],[1,-1]]
>>> iris=datasets.load_iris()
>>> X=iris.data
>>> Y=iris.target
>>> estimators={'k_means_3': KMeans(n_clusters=3),
	    'k_means_iris_8': KMeans(n_clusters=8),
	    'k_means_iris_bad_init': KMeans(n_clusters=3, n_init=1, init='random')}
>>> 
>>> 
>>> fignum=1
>>> for name, est in estimators.iteritems():
	fig=pl.figure(fignum,figsize=(4,3))
	pl.clf()
	ax=Axes3D(fig,rect=[0,0,.95,1], elev=48, azim=134)
	pl.cla()
	est.fit(X)
	labels=est.labels_
	ax.scatter(X[:,3],X[:,0],X[:,2], c=labels.astype(np.float16))
	ax.w_xaxis.set_set_ticklabels([])
	ax.w_yaxis.set_ticklabels([])
	ax.w_zaxis.set_ticklabels([])
	ax.set_xlabel('Petal width')
	ax.set_ylabel('Sepal length')
	ax.set_zlabel('Petal length')
	fignum=fignum+1


>>> for name, est in estimators.items():
	fig=pl.figure(fignum,figsize=(4,3))
	pl.clf()
	ax=Axes3D(fig,rect=[0,0,.95,1], elev=48, azim=134)
	pl.cla()
	est.fit(X)
	labels=est.labels_
	ax.scatter(X[:,3],X[:,0],X[:,2], c=labels.astype(np.float16))
	ax.w_xaxis.set_set_ticklabels([])
	ax.w_yaxis.set_ticklabels([])
	ax.w_zaxis.set_ticklabels([])
	ax.set_xlabel('Petal width')
	ax.set_ylabel('Sepal length')
	ax.set_zlabel('Petal length')
	fignum=fignum+1

	
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=3, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)

>>> for name, est in estimators.items():
	fig=pl.figure(fignum,figsize=(4,3))
	pl.clf()
	ax=Axes3D(fig,rect=[0,0,.95,1], elev=48, azim=134)
	pl.cla()
	est.fit(X)
	labels=est.labels_
	ax.scatter(X[:,3],X[:,0],X[:,2], c=labels.astype(np.float16))
	ax.w_xaxis.set_ticklabels([])
	ax.w_yaxis.set_ticklabels([])
	ax.w_zaxis.set_ticklabels([])
	ax.set_xlabel('Petal width')
	ax.set_ylabel('Sepal length')
	ax.set_zlabel('Petal length')
	fignum=fignum+1

	
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=3, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)
<mpl_toolkits.mplot3d.art3d.Path3DCollection object at 0x00000158188DA160>
[]
[]
[]
Text(0.5,0,'Petal width')
Text(0.5,0,'Sepal length')
Text(0.5,0,'Petal length')
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=8, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)
<mpl_toolkits.mplot3d.art3d.Path3DCollection object at 0x0000015818355550>
[]
[]
[]
Text(0.5,0,'Petal width')
Text(0.5,0,'Sepal length')
Text(0.5,0,'Petal length')
KMeans(algorithm='auto', copy_x=True, init='random', max_iter=300,
    n_clusters=3, n_init=1, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)
<mpl_toolkits.mplot3d.art3d.Path3DCollection object at 0x00000158183CBCF8>
[]
[]
[]
Text(0.5,0,'Petal width')
Text(0.5,0,'Sepal length')
Text(0.5,0,'Petal length')
>>> fig=pl.figure(fignum,figsize=(4,3))
>>> pl.clf()
>>> ax=Axes3D(fig, rect=[0,0,.95,1], elev=48, azim=134)
>>> pl.cla()
>>> for name, label in [('Setosa',0),
		    ('Versicolour',1),
		    ('Virginica',2)]:
	ac.text3D(X[Y==label,3].mean(),
		  X[y==label,0].mean()+1.5,
		  X[Y==label,2].mean(),name,
		  horizontalalignment='center',
		  bbox=dict(alpha=.5, edgecolor='w', facecolot='w')
>>> 
>>> 
>>> 
>>> 
>>> 
>>> for name, label in [('Setosa',0),
		    ('Versicolour',1),
		    ('Virginica',2)]:
	ax.text3D(X[Y==label,3].mean(),
		  X[Y==label,0].mean()+1.5,
		  X[Y==label,2].mean(),name,
		  horizontalalignment='center',
		  bbox=dict(alpha=.5, edgecolor='w', facecolot='w'))

>>> for name, label in [('Setosa',0),
		    ('Versicolour',1),
		    ('Virginica',2)]:
	ax.text3D(X[Y==label,3].mean(),
		  X[Y==label,0].mean()+1.5,
		  X[Y==label,2].mean(),name,
		  horizontalalignment='center',
		  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

	
Text(0.244,6.506,'Setosa')
Text(1.326,7.436,'Versicolour')
Text(2.026,8.088,'Virginica')
>>> 
>>> 
>>> 
>>> 
>>> 
>>> Y=np.choose(Y,[1,2,0]).astype(np.float)
>>> ax.scatter(X[:,3], X[:,0], X[:,2], c=Y)
<mpl_toolkits.mplot3d.art3d.Path3DCollection object at 0x00000158185B3470>
>>> ax.w_xaxis.set_ticklabels([])
[]
>>> ax.w_yaxis.set_ticklabels([])
[]
>>> ax.w_zaxis.set_ticklabels([])
[]
>>> ax.set_xlabel('Petal Width')
Text(0.5,0,'Petal Width')
>>> ax.set_ylabel('sepal length')
Text(0.5,0,'sepal length')
>>> ax.set_zlabel('petal length')
Text(0.5,0,'petal length')
>>> pl.show()
