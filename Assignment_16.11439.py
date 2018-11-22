
# coding: utf-8

# ## In this assignment I have to compress racoon grey scale image into 5 clusters. In the end, visualize both raw and compressed image and look for quality difference.
# <p>The raw image is available in spicy.misc package with the name face. </p>
# 

# ### Importing Modules

# In[1]:


import numpy as np
from sklearn import cluster, datasets
from scipy import misc
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# ### Loading Data

# In[2]:


#Get a 1024 x 768, gray image of a raccoon face.
face = misc.face(gray=True)


# ### Data Exploration/Analysis

# In[3]:


#Show gray image of raccoon face.
plt.gray()
plt.figure(figsize=(16,9))
plt.imshow(face)
plt.show()


# In[4]:


#Prints the shape of the face
print(face.shape)


# In[5]:


#Prints data type of face
print(face.dtype)


# ### Image Compression

# In[6]:


#Given, number of clusters = 5
n_clusters = 5

#Reshaping the image
image = face.reshape(face.shape[0]*face.shape[1],1)

#Initializing KMeans
kmeans= cluster.KMeans(n_clusters=n_clusters,n_init=10,max_iter=200, random_state=0)

#Fitting with input
kmeans.fit(image)


# In[7]:


#Rows and Columns size of the face image
rows = face.shape[0]
cols = face.shape[1]

#KMeans labels as array
labels = np.asarray(kmeans.labels_,dtype=np.uint8 ) 

#Reshaping the labels with shape of face image
face_compressed = labels.reshape(rows,cols)


# In[8]:


#Show gray image of compressed raccoon face image.
plt.gray()
plt.figure(figsize=(16,9))
plt.imshow(face_compressed)


# ### Compare Two Images

# In[9]:


#The 'Mean Squared Error' between the two images is the sum of the squared difference between the two images;
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
#Returns the MSE (The lower the error, the more "similar" the two images are)
    return err
 
def compare_images(imageA, imageB, title):
#Computes the mean squared error and structural similarity index for the images
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
 
    #Setup the figure
    fig = plt.figure(figsize=(16,9))
    plt.title(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))    
 
    #Shows first image
    ax = fig.add_subplot(1, 2, 1)    
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("off")
 
    #Shows the second image
    ax = fig.add_subplot(1, 2, 2)    
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("off")
 
    #Shows the images   
    plt.show()


# In[10]:


compare_images(face, face_compressed, "Comparison of Raw and Compressed racoon grey scale image")


# ### Quality difference

# In[11]:


print("Mean Squared Error:", mse(face, face_compressed))


# In[12]:


print("Structural Similarity Index:", ssim(face, face_compressed))

