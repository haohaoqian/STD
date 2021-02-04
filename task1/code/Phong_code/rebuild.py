import numpy as np
from render import render

def rebuild(images,lvectors,n_s,epochs=1000):

    alpha = np.zeros([168,168])
    beta = np.ones([168,168])
    s = np.ones([168,168])
    X = np.zeros([168,168])
    Y = np.zeros([168,168])
    Z = -np.ones([168,168])

    print('Training for {} epochs...'.format(epochs))
    for i in range(epochs):
        loss_sum=0
        g_alpha=np.zeros([168,168])
        g_beta=np.zeros([168,168])
        g_s=np.zeros([168,168])
        g_X=np.zeros([168,168])
        g_Y=np.zeros([168,168])
        g_Z=np.zeros([168,168])

        if i<50:lr=0.1
        elif i<500:lr=0.01
        else:lr=0.001

        for j in range(7):
            image=images[j,:,:]
            lvector=lvectors[j,:]

            pred,M,N=render(alpha,beta,s,X,Y,Z,n_s,lvector[np.newaxis,:],is_train=True)
            g_alpha+=np.clip(2*(image-pred),-1,1)
            g_beta+=np.clip(2*(image-pred)*M,-1,1)
            g_s+=np.clip(2*(image-pred)*np.power(N,n_s),-1,1)
            g_X+=np.clip(2*(image-pred)*(beta*lvector[0]-2*s*n_s*np.power(N,n_s-1)*(Z*lvector[0]-X*lvector[2])),-1,1)
            g_Y+=np.clip(2*(image-pred)*(beta*lvector[1]-2*s*n_s*np.power(N,n_s-1)*(Z*lvector[1]-Y*lvector[2])),-1,1)
            g_Z+=np.clip(2*(image-pred)*(beta*lvector[2]-2*s*n_s*np.power(N,n_s-1)*(X*lvector[0]+Y*lvector[1]+Z*lvector[2])),-1,1)
            loss=np.abs(image-pred)
            loss_sum+=np.mean(loss)

        alpha=alpha-g_alpha*lr/7
        beta=beta-g_beta*lr/7
        s=s-g_s*lr/7
        X=X-g_X*lr/7
        Y=Y-g_Y*lr/7
        Z=Z-g_Z*lr/7

        print('Training epoch {}|lr={}|Loss={}'.format(i+1,lr,loss_sum/7))

    vector=np.zeros([3,168,168])
    for j in range(168):
        for k in range(168):
            answer = np.linalg.solve(np.dot(lvectors.T,lvectors),np.dot(lvectors.T,images[:,j,k]))
            temp = np.sqrt(np.sum(answer**2))
            dataB = images[:,j,k]/temp+lvectors[:,-1]
            answer = np.linalg.solve(np.dot(lvectors[:,0:2].T,lvectors[:,0:2]),np.dot(lvectors[:,0:2].T,dataB))
            vector[0:2,j,k] = answer/np.sqrt(answer[0]**2+answer[1]**2+1)
            vector[-1,j,k]=-1/np.sqrt(answer[0]**2+answer[1]**2+1)

    return alpha,beta,s,X,Y,Z,vector