import pandas
import numpy
import math
import random

testPct = 0.25
numCVFolds = 5

# Read in data
filePath = r'C:\HighMileage\HM_w_Demo.csv'
demoDat = pandas.read_csv(filePath,
                          na_values={'SD_EDUCATION':'UNKNOWN',
                                     'PRESENCE_OF_CHILDREN':'U', 
                                     'MODEL':['MEDIUM HEAVY', 'MOTOR HOME', 'CROWN VICTORIA']}, 
                          parse_dates=['MAIL_DATE'])
demoDat.set_index(['CustomerID', 'LETTER_CODE', 'MAIL_DATE'], inplace=True)

# Convert catgorical fields to dummies
fields = demoDat.dtypes
objFields = fields[fields=='object'].index
demoDummied = pandas.get_dummies(demoDat, dummy_na=True, columns=objFields)

# remove fields with no missing values
del demoDummied['CASS_STATE_nan']
del demoDummied['CENSUS_DIVISION_nan']
del demoDummied['URBANIZATION_nan']
del demoDummied['MODEL_nan']

# Which fields have missing values
counts = demoDummied.count()
print('Numerical Fields with missing values:')
print(counts[counts<max(demoDummied.count())])
print('\n')

# Split into train/test
numRecs = demoDummied['RESPONSE'].count()
numRecsTest = math.floor(numRecs*testPct)
numRecsTrain = numRecs - numRecsTest

assignment = numRecsTest*['Test']
recsPerFold = math.ceil(numRecsTrain/5)
assignment.extend(recsPerFold*['Train Fold 1'])
assignment.extend(recsPerFold*['Train Fold 2'])
assignment.extend(recsPerFold*['Train Fold 3'])
assignment.extend(recsPerFold*['Train Fold 4'])
assignment.extend(recsPerFold*['Train Fold 5'])
random.shuffle(assignment)

# if doesn't split into an even number of records, remove records
while len(assignment)>numRecs:
    assignment.pop()

demoDummied['assignment'] = assignment
testCount = demoDummied[demoDummied['assignment']=='Test']['assignment'].count()
trainCount = demoDummied[demoDummied['assignment']!='Test']['assignment'].count()
print(demoDummied['assignment'].value_counts(dropna=False))

print('Splitting '+str(numRecs)+' records into a test set of '
      +str(testCount)+' records and a training set of '+
      str(trainCount))

print('Writing test file')
demoDummied[demoDummied['assignment']=='Test'].to_csv(r'C:\HighMileage\HM_Test.csv')
print('Writing training file')
demoDummied[demoDummied['assignment']!='Test'].to_csv(r'C:\HighMileage\HM_Train.csv')