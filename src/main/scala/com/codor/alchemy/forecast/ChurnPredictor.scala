package com.codor.alchemy.forecast

import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD.fromRDD
import org.apache.spark.api.java.JavaSparkContext.fromSparkContext
import org.apache.spark.rdd.RDD
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.api.Repartition
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import com.codor.alchemy.forecast.BaseParameterAveragingTrainingMaster
import org.deeplearning4j.spark.util.MLLibUtil
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions

import com.codor.alchemy.forecast.utils.Logs
import org.apache.spark.SparkContext
import org.nd4j.linalg.factory.Nd4j
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import com.codor.alchemy.forecast.utils.ReflectionsHelper
import org.nd4j.linalg.factory.Nd4j
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import com.codor.alchemy.forecast.AdvisorDataSetIterator

class ChurnPredictor(batchSize: Int = 50, featureSize: Int, nEpochs: Int, hiddenUnits: Int,
    miniBatchSizePerWorker: Int = 10, averagingFrequency: Int = 5, numberOfAveragings: Int = 3,
    learningRate: Double = 0.0018, l2Regularization: Double = 1e-5,
    dataDirectory: String, sc: SparkContext, exclude: RDD[String]) extends Serializable with Logs {

  //Nd4j.create(1)  
  ReflectionsHelper.registerUrlTypes()

  val conf = new NeuralNetConfiguration.Builder()
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(nEpochs) //
    .updater(Updater.RMSPROP)
    .regularization(true).l2(l2Regularization)
    .weightInit(WeightInit.XAVIER)
    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
    .gradientNormalizationThreshold(1.0)
    .learningRate(learningRate)
    .list()
    .layer(0, new GravesLSTM.Builder().nIn(featureSize).nOut(hiddenUnits) // 200
      .activation("relu").build()) // softsign relu
    .layer(1, new RnnOutputLayer.Builder().activation("identity") // relu
      .lossFunction(LossFunctions.LossFunction.MSE).nIn(hiddenUnits).nOut(1).build()) // MCXENT 200
    .pretrain(false).backprop(true).build()

  val net = new MultiLayerNetwork(conf)
  net.init()

  val tm = new BaseParameterAveragingTrainingMaster.Builder(5, 1) //ParameterAveragingTrainingMaster.Builder(5)
    .averagingFrequency(averagingFrequency)
    .batchSizePerWorker(miniBatchSizePerWorker)
    .saveUpdater(true)
    .workerPrefetchNumBatches(0)
    //.repartionData(Repartition.Always)
    .build();

  val sparkNetwork = new SparkDl4jMultiLayer(sc, net, tm)
  var instanceIndices: List[String] = null

  def train() = {
    val iterator = new AdvisorDataSetIterator(dataDirectory, batchSize, true)
    val buf = scala.collection.mutable.ListBuffer.empty[DataSet]
    while (iterator.hasNext)
      buf += iterator.next
    val rdd = sc.parallelize(buf)

    info("Examples: " + rdd.count() + " - " + rdd.first().getFeatureMatrix.columns() +
      rdd.first().getFeatureMatrix.columns() + rdd.first().getLabels.columns())

    import scala.collection.JavaConversions._
    instanceIndices = iterator.getAdvisors().toList

    info("RDD Train: " + rdd.count() + " - Row: " + rdd.first().getFeatureMatrix.rows()
      + " - Col: " + rdd.first().getFeatureMatrix.columns() + " - LRow: " + rdd.first().getLabels.rows()
      + " - LCol: " + rdd.first().getLabels.columns())

    info("Feature: " + rdd.first())

    instanceIndices = iterator.getAdvisors().toList

    sparkNetwork.fit(rdd)
  }

  def prepare(dataset: RDD[(String, Iterable[String])]) = {
    val predict = dataset.map(f => (f._1, f._2.filter { x => x.contains("201605") }))
      .map(f => (f._1, f._2.map { x => x.reverse.tail.reverse }))
    val training = dataset.subtract(predict)

    training.map { x => sc.parallelize(x._2.toList).repartition(1).saveAsTextFile(dataDirectory + "train/" + x._1) }
    predict.map { x => sc.parallelize(x._2.toList).repartition(1).saveAsTextFile(dataDirectory + "test/" + x._1) }

    this
  }

  def predict(network: MultiLayerNetwork): RDD[(String, String)] = {

    val iterator = new AdvisorDataSetIterator(dataDirectory, batchSize, false)
    val buf = scala.collection.mutable.ListBuffer.empty[DataSet]
    while (iterator.hasNext)
      buf += iterator.next
    val rdd = sc.parallelize(buf)

    info("RDD Predict: " + rdd.count() + " - Row: " + rdd.first().getFeatureMatrix.rows()
      + " - Col: " + rdd.first().getFeatureMatrix.columns() + " - LRow: " + rdd.first().getLabels.rows()
      + " - LCol: " + rdd.first().getLabels.columns())

    // info("Feature: " + rdd.first() + " - " + indices.count() + " - " + indices.first())

    val tm = new NewParameterAveragingTrainingMaster.Builder(5)
      .averagingFrequency(averagingFrequency)
      .batchSizePerWorker(miniBatchSizePerWorker)
      .saveUpdater(true)
      .workerPrefetchNumBatches(0)
      .repartionData(Repartition.Always)
      .build();

    //val trainedNetworkWrapper = new SparkDl4jMultiLayer(sc, network, tm)

    val indices = sc.parallelize(instanceIndices).zipWithIndex().map(f => (f._2, f._1))

    val examples = rdd.map { x =>
      //var features: List[Array[Double]] = List()
      //(0 until x.getFeatureMatrix.rows).foreach { i =>
      //  features = features :+ x.getFeatureMatrix.getRow(i).data().asDouble()
      //}
      val labels = x.getLabels.data().asDouble()
      //new LabeledPoint(labels(0), MLLibUtil.toVector(x.getFeatureMatrix.dup()))

      //
      //features.zip(labels).map(f => new LabeledPoint(f._2, Vectors.dense(f._1)))
      // }//.flatMap { x => x }

      // examples.map { point =>
      //val vector = MLLibUtil.toVector(x.getFeatureMatrix.dup())
      val score = network.output(x.getFeatureMatrix.dup()).data().asDouble() //(0)
      (labels.mkString(","), score.mkString(",")) //Array(labels.mkString(","), score.mkString(","))
    }

    //val examples = MLLibUtil.fromDataSet(sc, rdd).rdd

    //info("Examples: " + examples.count() + " - " + examples.first().toString())

    //val z = indices.join(examples.zipWithIndex().map(f => (f._2, f._1))).map(f => (f._2._1, f._2._2.mkString(",")))

    examples
  }

}
