from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('ml-wine').getOrCreate()
inputTrainDF = spark.read.option("delimiter", ";").csv('/data/train.csv', header = True, inferSchema = True)
inputTestDF = spark.read.option("delimiter", ";").csv('/data/test.csv', header = True, inferSchema = True)

from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# select the columns to be used as the features (all except `quality`)
featureColumns = [c for c in inputTrainDF.columns if c != 'quality']

# create and configure the assembler
assembler = VectorAssembler(inputCols=featureColumns, 
                            outputCol="features")

# transform the original data
#dataDF = assembler.transform(inputDF)


from pyspark.ml.evaluation import RegressionEvaluator

# create a regression evaluator with RMSE metrics

evaluator = RegressionEvaluator(
    labelCol='quality', predictionCol="prediction", metricName="rmse")

# split the input data into traning and test dataframes with 70% to 30% weights
#(trainingDF, testDF) = inputDF.randomSplit([0.7, 0.3])

from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor

# define the random forest estimator
rf = RandomForestRegressor(featuresCol="features", labelCol="quality", numTrees=100, maxBins=128, maxDepth=20, \
                           minInstancesPerNode=5, seed=33)
rfPipeline = Pipeline(stages=[assembler, rf])

# train the random forest model
rfPipelineModel = rfPipeline.fit(inputTrainDF)
# rfTrainingPredictions = rfPipelineModel.transform(inputTrainDF)
# rfTestPredictions = rfPipelineModel.transform(inputTestDF)
# print("Random Forest RMSE on traning data = %g" % evaluator.evaluate(rfTrainingPredictions))
# print("Random Forest RMSE on test data = %g" % evaluator.evaluate(rfTestPredictions))

rfPipelineModel.write().overwrite().save('output/rf.model')


# load the andom forest pipeline from the dist
from pyspark.ml import PipelineModel
loadedModel = PipelineModel.load('output/rf.model')
# loadedPredictionsDF = loadedModel.transform(inputTestDF)

# # evaluate the model again to see if we get the same performance
# print("Loaded model RMSE = %g" % evaluator.evaluate(loadedPredictionsDF))


predictions = loadedModel.transform(inputTestDF)
# labels_and_predictions = inputTestDF.map(lambda x: x.quality).zip(predictions)
# acc = labels_and_predictions.filter(lambda x: x[0] == x[1]).count() / float(inputTestDF.count())
# print("Model accuracy: %.3f%%" % (acc * 100))

# Evaluate the model. 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
print("Evaluating the model...")
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1 = evaluator.evaluate(predictions)
print("F1 = %g " % f1)
print("Model prediction finished... terminating.")



