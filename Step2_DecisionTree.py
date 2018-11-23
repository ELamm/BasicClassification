from datetime import datetime
import matplotlib.pyplot as plt
import pandas
import numpy
from sklearn import tree
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
import pickle
import graphviz

# helper function
def processFold(train, test, maxDepth=None, minSamplesLeaf=1):
    trainY = train.iloc[:, 0].values
    trainX = train.iloc[:, 1:].values
    testY = test.iloc[:, 0].values
    testX = test.iloc[:, 1:].values

    weightsTrain = {0: trainY.sum()/(trainY.shape[0]-trainY.sum()), 1:1}
 
    clf = tree.DecisionTreeClassifier(max_depth=maxDepth, min_samples_split=2,
                                      min_samples_leaf=minSamplesLeaf, class_weight=weightsTrain)
    clf.fit(trainX, trainY)
    trainYhat = clf.predict_proba(trainX)[:, 1]
    trainScore = roc_auc_score(trainY, trainYhat)
    testYhat = clf.predict_proba(testX)[:, 1]
    testScore = roc_auc_score(testY, testYhat)

    return [trainScore, testScore]

# read in training data
training = pandas.read_csv(r'C:\HighMileage\HM_Train.csv', parse_dates=['MAIL_DATE'])
training.set_index(['CustomerID', 'LETTER_CODE', 'MAIL_DATE'], inplace=True)
print('Training data read.')

# train with no parameters to rank full feature list
weightDict = {0:training['RESPONSE'].sum()/(training['RESPONSE'].count()-training['RESPONSE'].sum()), 
              1:1}

clf = tree.DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, class_weight=weightDict)
clf.fit(training.iloc[:, 1:-1].values, training.iloc[:, 0].values)

selectedFeats = set()
for i in range(0, len(clf.feature_importances_)):
    if clf.feature_importances_[i]>0:
        selectedFeats.add(training.columns[1+i])

F = open('Step2_TreeFullFeatures.pkl', 'wb')
pickle.dump(sorted(list(selectedFeats)), F)
F.close()
print('list of full feature importances saved to Step2_TreeFullFeatures.pkl')

# Train to find optimal tuning parameters
# tuning parameters
maxDepths = list(range(3, 10))
minSamplesPerLeaf = [30000, 25000, 20000, 15000, 10000, 5000, 2500, 1000, 500, 100]

# create a DataFrame to hold training results
resultInd = pandas.MultiIndex.from_product([maxDepths, minSamplesPerLeaf],
                                       names = ['max depth', 'min samples per leaf'])
resultDF = pandas.DataFrame(index=resultInd, 
                        columns=['training score 5', 'training score 4', 'training score 3', 
                                 'training score 2', 'training score 1', 'test score 5', 
                                 'test score 4', 'test score 3', 'test score 2', 'test score 1', 
                                 'avg training score', 'avg test score'])

# train all parameter combinations
for fold in [1, 2, 3, 4, 5]:
    train = training[training['assignment']!='Train Fold '+str(fold)].iloc[:, 0:-1]
    test = training[training['assignment']=='Train Fold '+str(fold)].iloc[:, 0:-1]
    print('CV fold '+str(fold)+' running at: '+str(datetime.now()))
    for depth in maxDepths:
        for minLeaf in minSamplesPerLeaf:
            result = processFold(train, test, maxDepth=depth, minSamplesLeaf=minLeaf)
            resultDF.loc[depth, minLeaf]['training score '+str(fold)] = result[0]
            resultDF.loc[depth, minLeaf]['test score '+str(fold)] = result[1]
print('all parameter test combinations and CV folds finished running at '+str(datetime.now()))

resultDF['avg training score'] = (resultDF['training score 5']+resultDF['training score 4']+
                                  resultDF['training score 3']+resultDF['training score 2']+
                                  resultDF['training score 1'])/5
resultDF['avg test score'] = (resultDF['test score 5']+resultDF['test score 4']+
                              resultDF['test score 3']+resultDF['test score 2']+
                              resultDF['test score 1'])/5

# save results
resultDF.to_csv(r'C:\HighMileage\Step1_resultDF.csv')
print(r'Cross Validation results saved to file: C:\HighMileage\Step1_resultDF.csv at '+str(datetime.now()))

# values for tuning parameters are where test score is the highest
fitParams = resultDF[resultDF['avg test score']==resultDF['avg test score'].max()].index.values
maxDepthFinal = fitParams[0][0]
minSamplesLeafFinal = fitParams[0][1]
print('Tuned value for maximum depth is '+str(maxDepthFinal)+'.')
print('Tuned value for minimum samples leaf is '+str(minSamplesLeafFinal)+'.')

# fitting curves
# show complexity low -> high
trainingY1 = resultDF['avg training score'].loc[maxDepthFinal, :].values
testY1 = resultDF['avg test score'].loc[maxDepthFinal, :].values
X1 = sorted(list(set(resultDF.index.get_level_values(1))))
X1.reverse()

trainingY2 = resultDF['avg training score'].swaplevel('max depth', 'min samples per leaf').loc[minSamplesLeafFinal, :]
testY2 = resultDF['avg test score'].swaplevel('max depth', 'min samples per leaf').loc[minSamplesLeafFinal, :]
X2 = sorted(list(set(resultDF.index.get_level_values(0))))

fig, ax1 = plt.subplots(1, 1)
ax1.plot([str(x) for x in X1], trainingY1, label='training AUC')
ax1.axvline(x=str(minSamplesLeafFinal), color='black', linestyle='--')
ax1.plot([str(x) for x in X1], testY1, label='test AUC')
ax1.set_title('Minimum Samples per leaf,\n max depth='+str(maxDepthFinal))
ax1.set_ylabel('AUC')
ax1.set_xlabel('Minimum Samples per Leaf')
plt.xticks(rotation='vertical')
plt.legend(loc='best')
fig.subplots_adjust(bottom=0.25)
fig.savefig('fittingCurve1.png')
print('Fitting curve for minimum samples per leaf written to fittingCurve1.png')

fig, ax2 = plt.subplots(1, 1)
ax2.plot(X2, trainingY2, label='training AUC')
ax2.axvline(x=maxDepthFinal, color='black', linestyle='--')
ax2.plot(X2, testY2, label='test AUC')
ax2.set_title('Max depth, minimum samples\n per leaf='+str(minSamplesLeafFinal))
ax2.set_ylabel('AUC')
ax2.set_xlabel('Maximum Depth')
plt.legend(loc='best')
fig.subplots_adjust(bottom=0.15)
fig.savefig('fittingCurve2.png')
print('Fitting curve for max depth written to fittingCurve2.png')

# train final classification tree on all training data
weightDict = {0:training['RESPONSE'].sum()/(training['RESPONSE'].count()-training['RESPONSE'].sum()), 
              1:1}

clf = tree.DecisionTreeClassifier(max_depth=maxDepthFinal, min_samples_leaf=minSamplesLeafFinal, 
                                  class_weight=weightDict)
clf.fit(training.iloc[:, 1:-1].values, training.iloc[:, 0].values)

selectedFeats = set()
for i in range(0, len(clf.feature_importances_)):
    if clf.feature_importances_[i]>0:
        selectedFeats.add(training.columns[1+i])

# save features used in final model
F = open('Step2_TreeFeatures.pkl', 'wb')
pickle.dump(sorted(list(selectedFeats)), F)
F.close()
print('list of features selected for final model saved to Step2_TreeFeatures.pkl')

# train with reduced feature set
training = training.reindex(columns=['RESPONSE']+sorted(list(selectedFeats)))
clf = tree.DecisionTreeClassifier(max_depth=maxDepthFinal, min_samples_leaf=minSamplesLeafFinal, 
                                  class_weight=weightDict)
clf.fit(training.iloc[:, 1:].values, training.iloc[:, 0].values)

# model persistence
joblib.dump(clf, 'DecisionTreeFinal.joblib')
print('final tree model saved to DecisionTreeFinal.joblib')

# write tree visual to pdf
dot_data = tree.export_graphviz(clf, out_file=None, feature_names = training.columns[1:],
                                class_names=['no response', 'response'], filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("HM_DecisionTree_Weighted")
print('visual depiction of final decision tree saved to HM_DecisionTree_Weighted.pdf')