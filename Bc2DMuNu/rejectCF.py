#To Do:
#   - finish function (future wrapper) to plot selected branches with LaTex axes/title (with units) and **kwd, *args 
#     for range etc
#   - setup class and inheritance to use this plotter from executable/module or alike
#   - change name of branches and implement latex in plots
#   - feature ranking
from root_numpy import tree2array, root2array, rec2array
import numpy as np
import ROOT

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neural_network import MLPClassifier

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pandas.core.common as com
from pandas.core.index import Index
from pandas.tools import plotting
from sklearn.metrics import roc_curve, auc

import time
start_time = time.time()





#convert skimmed MC trees to numpy arrays
MCdir = '/var/pcfst/r01/lhcb/delaney/Analysis/LHCbAnalysis/Bc2Dmunu/sub_jobs_MC_SB/'
BkgMC='MCBu2DMuNu.root'
SigMC='MCBc2DMuNu.root'

bkgFile = ROOT.TFile(MCdir+BkgMC)
sigFile = ROOT.TFile(MCdir+SigMC)

bkgTree = bkgFile.Get('AnalysisTree')
sigTree = sigFile.Get('AnalysisTree')


#eventually load one tree and include itype selection
Sevtmax=16000
Bevtmax=16000


#should rename with string manipulation
#'B_plus_MCORRERR/B_plus_MCORR' bdt output correlated with MCORR
branches=['TMath::Log(B_plus_PT)', 
'TMath::Log(B_plus_IPCHI2_OWNPV)', 'TMath::ACos(B_plus_DIRA_OWNPV)', 'B_plus_LTIME', 'TMath::Log(D0_PT)','TMath::Log(D0_IPCHI2_OWNPV)',
'TMath::Log(Mu_plus_PT)', 'TMath::Log(Mu_plus_MIPCHI2PV)'] 

#implement cut on lifetime for sensible values
sigArray = tree2array(sigTree, branches, selection='B_plus_LTIME>0', start=0, stop=Sevtmax) 
sig=rec2array(sigArray)

bkgArray = tree2array(bkgTree, branches, selection='B_plus_LTIME>0', start=0, stop=Bevtmax)
bkg=rec2array(bkgArray)


print '\n--- Selected %i entries to read in from MC ---\n'%(len(sig)+len(bkg))
print '--- %s signal events and %s bkg events ---\n'%(len(sig),len(bkg))


#scikit learn requires 2D array (n_samples, n_features)
X = np.concatenate((sig, bkg))
y = np.concatenate((np.ones(sig.shape[0]),
                    np.zeros(bkg.shape[0])))

##stack branches and the y-values; y values tag S/B
#df = pd.DataFrame(np.hstack((X,y.reshape(y.shape[0], -1))),
#                  columns=branches+['y'])
#
#
##plot branches, turn this into func() with wrapper for preselection
#for branch in branches:
#    BcMC = df[(df['y']==1)]['%s'%branch]
#    BuMC = df[(df['y']==0)]['%s'%branch]
#    canvas = plt.figure()
#    plt.title('%s'%branch)
#    ax = canvas.add_subplot(111)
#    _ = ax.hist(BuMC.values,  normed=True, bins=100, label='MC Bu2DMuNu', fill=True  , stacked=True, histtype='step', 
#                    edgecolor='darkblue', linewidth=1, alpha=.7) 
#    _ = ax.hist(BcMC.values, color='lightcoral',  normed=True, bins=100, label='MC Bc2DMuNu',
#                fill=True , stacked=True, histtype='step', edgecolor='maroon', linewidth=1, alpha=.7)
#    _ = ax.legend(loc='best')
#    plt.savefig('PRESENTATION/%s.pdf'%branch)
#
#bg = df.y == 0
#sig = df.y == 1
#
#def correlations(data, title, **kwds):
#    """Calculate pairwise correlation between features.
#    
#    Extra arguments are passed on to DataFrame.corr()
#    """
#    # simply call df.corr() to get a table of
#    # correlation values if you do not need
#    # the fancy plotting
#    corrmat = data.corr(**kwds)
#
#    fig, ax1 = plt.subplots(ncols=1, figsize=(10,8))
#    
#    opts = {'cmap': plt.get_cmap("RdBu"),
#            'vmin': -1, 'vmax': +1}
#    heatmap1 = ax1.pcolor(corrmat, **opts)
#    plt.colorbar(heatmap1, ax=ax1)
#
#    ax1.set_title('Input Variable Correlations')
#
#    labels = corrmat.columns.values
#    for ax in (ax1,):
#        # shift location of ticks to center of the bins
#        ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
#        ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
#        ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
#        ax.set_yticklabels(labels, minor=False)
#        
#    plt.tight_layout()
#    plt.savefig('PRESENTATION/%s.pdf'%title)
#    
## remove the y column from the correlation matrix
## after using it to select background and signal
#correlations(df[bg].drop('y', 1), 'Bkg_correlatios')
#correlations(df[sig].drop('y', 1),'Sig_correlations')


#------------------------------------------------------------------------------------------------
##split data into dev and eval samples
X_dev,X_eval, y_dev,y_eval = train_test_split(X, y, test_size=.4, random_state=23)
#X_train,X_test, y_train,y_test = train_test_split(X_dev, y_dev, test_size=0.4, random_state=199)
#------------------------------------------------------------------------------------------------

results = {}

def optimize_performance(clf, pars):
    
    def grid_search(clf, pars):
        
        grid = GridSearchCV(clf, 
                            param_grid=pars, 
                            n_jobs=-1, 
                            scoring='roc_auc',
                            cv=3)
        grid.fit(X_dev, y_dev)
        return grid

    grid=grid_search(clf, pars)
     
    print "Best parameter set found on development set:"
    print
    print grid.best_estimator_
    print
    print "Grid scores on a subset of the development set:"
    print
    for params, mean_score, scores in grid.grid_scores_:
        print "%0.4f (+/-%0.04f) for %r"%(mean_score, scores.std(), params)
    print
    print "With the model trained on the full development set:"

    y_true, y_pred = y_dev, grid.decision_function(X_dev)
    print "  It scores %0.4f on the full development set"%roc_auc_score(y_true, y_pred)
    y_true, y_pred = y_eval, grid.decision_function(X_eval)
    print "  It scores %0.4f on the full evaluation set"%roc_auc_score(y_true, y_pred)
  

    #return grid.best_score_, grid.best_estimator_
    results[grid.best_estimator_] = grid.best_score_
    #clf = sorted(results.iteritems(), key=lambda (k,v): (v,k), reverse=True)[0][0]
    
    return grid.best_estimator_

gbt = GradientBoostingClassifier()

gbt_param_grid = {'n_estimators': [50,200,400,1000],
                  'max_depth': [1, 3, 8],
                  'learning_rate': [.1, .2,  1.]}




gbt = optimize_performance(gbt, gbt_param_grid)
#print 'Checking best classifier is being seleted', gbt

#clfs=[]

#gbt_params = ((.1,3), (.1,5), (.1,1 ), 
#              (.2,3), (.2,5), (.2,1), 
#              (1.,3), (1.,5), (1.,1))


#clf=GradientBoostingClassifier(max_depth=gbt.max_depth,
#                                   learning_rate=gbt.learning_rate,
#                                   n_estimators=1000)
#clf.fit(X_train, y_train) #train
#clfs.append(clf)
#from sklearn.metrics import roc_curve, auc
#
#
#def validation_curve(clfs, train, test):
#    plt.figure()
#    X_test, y_test = test
#    X_train, y_train = train
#    
#    for n,clf in enumerate(clfs):
#        
#        #for every estimator save train and test score
#        test_score = np.empty(len(clf.estimators_))
#        train_score = np.empty(len(clf.estimators_))
#
#        #compute the score at every iteration
#        for i, pred in enumerate(clf.staged_decision_function(X_test)):
#            test_score[i] = 1- roc_auc_score(y_test, pred)
#
#        for i, pred in enumerate(clf.staged_decision_function(X_train)):
#            train_score[i] = 1-roc_auc_score(y_train, pred)
#        
#
#        #plot vertical line at best score location
#        
#        best_iter = np.argmin(test_score)
#        print '-------- best iteration at %s for %s'%(best_iter, clf)
#        if clf.learning_rate==gbt.learning_rate and clf.max_depth==gbt.max_depth:
#                print 'before check for overtraining, best gbt had %i estimators'%gbt.n_estimators
#                gbt.n_estimators=best_iter
#                print 'Update: best gbt now haa %i number of estimators'%gbt.n_estimators
#
#        learn = clf.get_params()['learning_rate']
#        depth = clf.get_params()['max_depth']
#        
#        test_line = plt.plot(
#                    test_score,
#                    label='learn=%.1f depth=%i (%.2f)'%(learn,depth, test_score[best_iter])
#                    )
#        
#        colour = test_line[-1].get_color()
#        plt.plot(train_score, '--', color=colour)
#        
#        plt.xlabel("Number of boosting iterations")
#        plt.ylabel("1 - area under ROC")
#        plt.axvline(x=best_iter, color=colour)
#        
#    plt.legend(loc='best')
#    plt.savefig('PRESENTATION/validation.pdf')
#    plt.show()

#validation_curve(clfs,
#                 (X_train,y_train),
#                 (X_test,y_test))

#print 'updated best candidate: ', gbt


#------------------------------------------------------------------------------------------------
#fit data and evaluate performance
#gbt.fit(X_train, y_train)
#gbty_predicted = gbt.predict(X_test)
#gbtdecision = gbt.decision_function(X_test)
#
#print classification_report(y_test, gbty_predicted,
#                                    target_names=["background", "signal"])
#print "Area under ROC curve: %.4f"%(roc_auc_score(y_test, gbtdecision))
#
#
#gbtfpr, gbttpr, gbtthresholds = roc_curve(y_test, gbtdecision)
#gbtroc_auc = auc(gbtfpr, gbttpr)
#
#plt.figure()
#plt.plot(gbtfpr, gbttpr, lw=1, label='MVA ROC (area = %0.2f)'%(gbtroc_auc))
#plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
#plt.xlim([-0.05, 1.05])
#plt.ylim([-0.05, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic')
#plt.legend(loc="lower right")
#plt.savefig('PRESENTATION/ROC.pdf')
#plt.show()

def compare_train_test(clf, X_dev, y_dev, X_eval, y_eval, bins=50):
    decisions = []
    for X,y in ((X_dev, y_dev), (X_eval, y_eval)):
        d1 = clf.decision_function(X[y>0.5]).ravel()
        d2 = clf.decision_function(X[y<0.5]).ravel()
        decisions += [d1, d2]
        
    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low,high)
    
    plt.hist(decisions[0],
             color='r', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='S (train)')
    plt.hist(decisions[1],
             color='b', alpha=0.5, range=low_high, bins=bins,
             histtype='stepfilled', normed=True,
             label='B (train)')

    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale
    
    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')
    
    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

    plt.xlabel("BDT output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc='best')
    plt.savefig('futurerun_gbt_FULLSAMPLE_novalidation_bdtoutput.pdf')
    plt.show()

compare_train_test(gbt, X_dev, y_dev, X_eval, y_eval)

print("--- Run time = %s seconds ---" % (time.time() - start_time))
