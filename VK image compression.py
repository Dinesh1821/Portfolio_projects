#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# In[8]:


def open(path):
    Im=Image.open(path)
    im=np.array(Im)
    aRed=im[:,:,0]
    aGreen=im[:,:,1]
    aBlue=im[:,:,2]
    print("no.of singular values: %d"%(len(aRed)))
    return [aRed,aGreen,aBlue,Im]


# In[20]:


def compress(Datamatrix,r):
    U,D,VT=np.linalg.svd(Datamatrix)
    D=np.diag(D)
    Xapprox=U[:,0:r]@D[0:r,0:r]@VT[0:r,:]
    cumsum=np.cumsum(np.diag(D))/np.sum(np.diag(D))
    Xapprox=Xapprox.astype('uint8')
    return [Xapprox,cumsum,D]


# In[21]:


filename=input("Enter file name:")
extension=input("enter extension")
file=filename+extension


# In[22]:


r=int(input())
aRed,aGreen,aBlue,imorg=open(file)


# In[23]:


aRedcomp,cumsumred,diagRed=compress(aRed,r)
aGreencomp,cumsumgreen,diaggreen=compress(aGreen,r)
aBluecomp,cumsumblue,diagblue=compress(aBlue,r)


# In[24]:


plt.plot((cumsumred+cumsumgreen+cumsumblue)/3,'k-o')


# In[26]:


plt.semilogy(np.diag(diagRed+diaggreen+diagblue),'k-o')


# In[29]:


imr=Image.fromarray(aRedcomp,mode=None)
img=Image.fromarray(aGreencomp,mode=None)
imb=Image.fromarray(aBluecomp,mode=None)


# In[30]:


newimage=Image.merge("RGB",(imr,img,imb))
newimage.show()


# In[ ]:




