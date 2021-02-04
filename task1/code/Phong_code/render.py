import numpy as np

def render(alpha,beta,s,X,Y,Z,n_s,lvectors,is_train=False):

    n=np.concatenate((X[np.newaxis,:,:],Y[np.newaxis,:,:],Z[np.newaxis,:,:]),axis=0)
    X_m=-2*X*Y
    Y_m=-2*Y*Z
    Z_m=-np.square(Z)+np.square(X)+np.square(Y)
    v=np.concatenate((X_m[np.newaxis,:,:],Y_m[np.newaxis,:,:],Z_m[np.newaxis,:,:]),axis=0)

    M=np.einsum('ij,jkl->ikl',lvectors,n)
    N=np.einsum('ij,jkl->ikl',lvectors,v)

    if is_train:
        imgs=alpha+beta*M-s*np.power(N,n_s)
        img=np.squeeze(imgs,axis=0)
        M=np.squeeze(M,axis=0)
        N=np.squeeze(N,axis=0)
        return img,M,N
    else:
        imgs=np.clip(alpha+beta*M-s*np.power(N,n_s),0,255)
        return imgs