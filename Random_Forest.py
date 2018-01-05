
# coding: utf-8

# In[66]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import matplotlib.cm as cm
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier


# In[19]:

mona = Image.open("monaLisa.jpg") ; mona = np.asarray(mona)
plt.imshow(mona) ; plt.xticks([], []) ; plt.yticks([], []) ;
plt.xlabel("Mona Lisa")
plt.show()


# ## Preprocessing the input. To build your “training set,” uniformly sample 5,000 random (x, y) coordinate locations.

# In[31]:

ROW, COL, _ = mona.shape

def getSamples(size = 5000):
    data, labels = [], []
    for i in range(size):
        a = np.random.randint(ROW)
        b = np.random.randint(COL)
        data.append((a, b))
        label = mona[a][b]
        labels.append(label)
    data = np.array(data) ; labels = np.array(labels)
    return data, labels

data, labels = getSamples(5000)


# In[29]:

data


# In[11]:

labels


# ## Preprocessing the output. Sample pixel values at each of the given coordinate locations. Each pixel contains red, green, and blue intensity values, so decide how you want to handle this.

# In[81]:

processed = np.array(mona, dtype=float)
for r in range(ROW):
    for c in range(COL):
        processed[r,c] = mona[r,c] / 255

processedLabel = np.array(labels, dtype=float)
processedLabel = processedLabel / 255        

fig, ax = plt.subplots(nrows = 1, ncols = 2)
plt.subplot(1,2,1)
plt.imshow(mona)
plt.xlabel("Original")
plt.xticks([], []) ; plt.yticks([], []) ;

plt.subplot(1,2,2)
plt.imshow(processed)
plt.xticks([], []) ; plt.yticks([], []) ;
plt.xlabel("Scaled Image")
plt.savefig("images/scalePixel.png")
plt.show()


# In[42]:

def rfPredictions(points, labels, depth = None, n = 1):
    m = RandomForestRegressor(max_depth = depth, n_estimators = n)
    m.fit(points, labels)
    predictions = np.zeros([ROW, COL,3])
    for i in range(ROW):
        for j in range(COL):
            loc = [i,j]
            loc = np.array(loc)
            predictions[i,j] = m.predict(loc.reshape(1,-1)) / 255
    return predictions


# In[57]:

def plotRF(p, title) :
    plt.imshow(p)
    plt.xticks([], []) ; plt.yticks([], []) ;
    plt.title(title)
    plt.savefig("images/" + title + ".png")
    plt.show()


# In[43]:

rf0 = rfPredictions(data, labels)


# In[69]:

rf1 = rfPredictions(data, labels, depth = 1)


# In[70]:

plotRF(rf1, "Random Forest with Depth 1")


# In[51]:

rf2 = rfPredictions(data, labels, depth = 2)
rf3 = rfPredictions(data, labels, depth = 3)


# In[58]:

plotRF(rf2, "Random Forest with Depth 2")
plotRF(rf3, "Random Forest with Depth 3")


# In[53]:

rf5 = rfPredictions(data, labels, depth = 5)
rf10 = rfPredictions(data, labels, depth = 10)
rf15 = rfPredictions(data, labels, depth = 15)


# In[59]:

plotRF(rf5, "Random Forest with Depth 5")
plotRF(rf10, "Random Forest with Depth 10")
plotRF(rf15, "Random Forest with Depth 15")


# In[60]:

rf7t1 = rfPredictions(data, labels, depth = 7, n = 1)
rf7t3 = rfPredictions(data, labels, depth = 7, n = 3)


# In[61]:

plotRF(rf7t1, "Random Forest with 1 Tree")
plotRF(rf7t3, "Random Forest with 3 Trees")


# In[62]:

rf7t5 = rfPredictions(data, labels, depth = 7, n = 5)
rf7t10 = rfPredictions(data, labels, depth = 7, n = 10)


# In[63]:

plotRF(rf7t5, "Random Forest with 5 Trees")
plotRF(rf7t10, "Random Forest with 10 Trees")


# In[64]:

rf7t100 = rfPredictions(data, labels, depth = 7, n = 100)


# In[65]:

plotRF(rf7t100, "Random Forest with 100 Trees")


# In[ ]:

# K-NN
def knn(xlabel="KNN"):
    knn = KNeighborsClassifier(neighbors=1)
    knn.fit(data, labels)
    pred = np.zeros([ROW,COL,3])
    for i in range(ROW):
        for j in range(COL):
            point = [i,j]
            point = np.array(point)
            pred[i,j] = knn.predict(point.reshape(1,-1)) / 255
    return pred
pred = knn()


# In[78]:

plt.imshow(pred)
plt.xlabel('KNN')
plt.xticks([], []) ; plt.yticks([], []) ;
plt.savefig("images/KNN Mona Lisa.png")
plt.show()

