package com.gale.alchemy.forecast

import org.apache.spark.SparkConf
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.api.java.JavaSparkContext.toSparkContext
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
import org.deeplearning4j.spark.util.MLLibUtil
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions

import com.gale.alchemy.forecast.utils.Logs

class ChurnPredictor(batchSize: Int = 50, featureSize: Int, nEpochs: Int, hiddenLayers: Int,
    miniBatchSizePerWorker: Int = 10, averagingFrequency: Int = 5, numberOfAveragings: Int = 3,
    learningRate: Double = 0.0018, l2Regularization: Double = 1e-5,
    dataDirectory: String) extends Serializable with Logs {

  val conf = new NeuralNetConfiguration.Builder()
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(nEpochs) //
    .updater(Updater.RMSPROP)
    .regularization(true).l2(l2Regularization)
    .weightInit(WeightInit.XAVIER)
    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
    .gradientNormalizationThreshold(1.0)
    .learningRate(learningRate)
    .list()
    .layer(0, new GravesLSTM.Builder().nIn(featureSize).nOut(hiddenLayers) // 200
      .activation("softsign").build()) // softsign
    .layer(1, new RnnOutputLayer.Builder().activation("relu") // relu
      .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(hiddenLayers).nOut(1).build()) // 200
    .pretrain(true).backprop(true).build()

  val net = new MultiLayerNetwork(conf)
  net.init()

  val sparkConf = new SparkConf()
  sparkConf.setMaster("local[" + 8 + "]")
  sparkConf.setAppName("ChurnPredictor")
  sparkConf.set("AVERAGE_EACH_ITERATION", String.valueOf(true))
  val sc = new JavaSparkContext(sparkConf)

  val tm = new ParameterAveragingTrainingMaster.Builder(5)
    .averagingFrequency(averagingFrequency)
    .batchSizePerWorker(miniBatchSizePerWorker)
    .saveUpdater(true)
    .workerPrefetchNumBatches(0)
    .repartionData(Repartition.Always)
    .build();

  val sparkNetwork = new SparkDl4jMultiLayer(sc, net, tm)

  def train() = {
    val iterator = new AdvisorDataSetIterator(dataDirectory, batchSize, true)
    val buf = scala.collection.mutable.ListBuffer.empty[DataSet]
    while (iterator.hasNext)
      buf += iterator.next
    val rdd = sc.parallelize(buf)

    sparkNetwork.fit(rdd)
  }

  def predict(): RDD[Array[Double]] = {
    val iterator = new AdvisorDataSetIterator(dataDirectory, batchSize, false)
    val buf = scala.collection.mutable.ListBuffer.empty[DataSet]
    while (iterator.hasNext)
      buf += iterator.next
    val rdd = sc.parallelize(buf)

    rdd.map { x => sparkNetwork.predict(MLLibUtil.toMatrix(x.getFeatureMatrix)).toArray }
  }
}