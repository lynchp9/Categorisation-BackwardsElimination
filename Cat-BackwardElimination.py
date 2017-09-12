List = np.array([0,0,0])
List = List.astype(float)
X_atrain = X_train
X_atest = X_test
if len(X_train.T) > 1:
    for i in range(0,len(X_atrain.T)-1):
        a = np.array([0,0,0])
        a = a.astype(float)
        X_ptrain = X_atrain
        X_ptest  = X_atest
       
        for j in range(0,len(X_ptrain.T)-1):
            X_ptrain = X_atrain
            X_ptest  = X_atest
            X_ptrain = np.delete(X_ptrain, [j], axis=1)
            X_ptest = np.delete(X_ptest, [j], axis=1)
            #Classifer
            from sklearn.ensemble import RandomForestClassifier
            classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
            classifier.fit(X_ptrain, y_train)
            y_pred = classifier.predict(X_ptest)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            from sklearn.metrics import precision_recall_fscore_support
            y_true = y_test
            Eval = precision_recall_fscore_support(y_true, y_pred, average = 'weighted')[2]
            
            if Eval >= a[0]:
                a[0] = Eval
                a[1] = j
            X_ptrain = X_atrain
            X_ptest =X_atest
        X_atrain = np.delete(X_atrain, [a[1].astype(int)],axis = 1)
        X_atest = np.delete(X_atest, [a[1].astype(int)],axis = 1)
        List = np.vstack([List, a])
        
List[0, :] = List[1, :]
X_ntrain = X_train
X_ntest = X_test
for l in range(1,np.argmax(List[:,0] + 1)):
        X_ntrain = np.delete(X_ntrain, [List[l,1].astype(int)],axis = 1)
        X_ntest = np.delete(X_ntest, [List[l,1].astype(int)],axis = 1)
        
X_train = X_ntrain
X_test = X_ntest