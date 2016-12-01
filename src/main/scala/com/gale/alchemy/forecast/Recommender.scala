package com.gale.alchemy.forecast

import org.deeplearning4j.nn.conf.BackpropType
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.GravesLSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.api.Repartition
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph
import org.deeplearning4j.spark.impl.graph.dataset.DataSetToMultiDataSetFn
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.util.MLLibUtil
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import com.gale.alchemy.forecast.utils.ReflectionsHelper
import com.gale.alchemy.forecast.utils.Logs
import org.nd4j.linalg.dataset.api.MultiDataSet

class Recommender(batchSize: Int = 50, featureSize: Int, nEpochs: Int, hiddenLayers: Int,
    miniBatchSizePerWorker: Int = 10, averagingFrequency: Int = 5, numberOfAveragings: Int = 3,
    learningRate: Double = 0.2, l2Regularization: Double = 0.001, labelSize: Int,
    dataDirectory: String, sc: SparkContext) extends Serializable with Logs {
 
  ReflectionsHelper.registerUrlTypes()  

  val tm = new BaseParameterAveragingTrainingMaster.Builder(5, 1)
    .averagingFrequency(averagingFrequency)
    .batchSizePerWorker(miniBatchSizePerWorker)
    .saveUpdater(true)
    .workerPrefetchNumBatches(0)
    .repartionData(Repartition.Always)
    .build();

  /*val conf = new NeuralNetConfiguration.Builder()
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(nEpochs)
    .updater(Updater.RMSPROP)
    .regularization(true).l2(l2Regularization)
    .weightInit(WeightInit.XAVIER)
    //.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
    //.gradientNormalizationThreshold(1.0)
    .learningRate(learningRate)
    .list()
    .layer(0, new GravesLSTM.Builder().nIn(featureSize).nOut(hiddenLayers).dropout(0.5)
      .activation("relu").build())
    .layer(1, new RnnOutputLayer.Builder().activation("softmax")
      .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(hiddenLayers).nOut(111).build())
    .pretrain(false).backprop(true).build()

  val net = new MultiLayerNetwork(conf)
  net.init()

  val sparkNetwork = new SparkDl4jMultiLayer(sc, net, tm) */

  val conf = new NeuralNetConfiguration.Builder()
    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    .learningRate(learningRate)
    .regularization(true).l2(l2Regularization)
    .weightInit(WeightInit.XAVIER)
    //.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
    //.gradientNormalizationThreshold(1.0)
    .updater(Updater.RMSPROP)
    .iterations(nEpochs)
    .seed(12345)
    .graphBuilder()
    .addInputs("input")
    .addLayer("firstLayer", new GravesLSTM.Builder().nIn(featureSize).nOut(hiddenLayers) // 200
      .activation("relu").build(), "input")
    .addLayer("secondLayer", new GravesLSTM.Builder().nIn(hiddenLayers).nOut(featureSize) // 200
      .activation("relu").build(), "firstLayer")
    .addLayer("outputLayer", new RnnOutputLayer.Builder().activation("softmax") // relu
      .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(featureSize).nOut(labelSize).build(), "secondLayer")
    .setOutputs("outputLayer")
    .backpropType(BackpropType.TruncatedBPTT)
    //.tBPTTForwardLength(4)
    //.tBPTTBackwardLength(4)
    .pretrain(false).backprop(true)
    .build()

  val sparkNet = new SparkComputationGraph(sc, conf, tm)
  sparkNet.setCollectTrainingStats(true) // for debugging and optimization purposes

  def train() = {
    //val iterator: AdvisorDataSetIterator =
    //  new AdvisorDataSetIterator(dataDirectory, batchSize, false, featureSize, labelSize) // true
    //val buf = scala.collection.mutable.ListBuffer.empty[DataSet]
  
    val iterator: AdvisorMDSIterator =
      new AdvisorMDSIterator(dataDirectory, batchSize, true, featureSize, labelSize)

    val buf = scala.collection.mutable.ListBuffer.empty[MultiDataSet]
  
    while (iterator.hasNext) {
      buf += iterator.next
    }
    val rdd = sc.parallelize(buf)

    info("Examples: " + rdd.count())
    
    sparkNet.fitMultiDataSet(rdd)	

  }

  def predict(graph: ComputationGraph, items: List[String]): RDD[List[(String, Double)]] = {

    val iterator: AdvisorMDSIterator =
      new AdvisorMDSIterator(dataDirectory, batchSize, false, featureSize, labelSize)
    val buf = scala.collection.mutable.ListBuffer.empty[MultiDataSet]

    while (iterator.hasNext) {
      val mds = iterator.next
      buf += mds
    }
    val rdd = sc.parallelize(buf)

    val predictions = rdd.map { x =>
      val score = graph.output(x.getFeatures(0).dup()).apply(0).data().asDouble()
      val outSum = score.take(111).sum // take just the real output
      val recalibrated = score.map { x => x * score.sum / outSum }
      items.zip(recalibrated)
    }

    predictions
  }

  def predict(network: MultiLayerNetwork, items: List[String]): RDD[List[(String, Double)]] = {
    import scala.collection.JavaConversions._
    val iterator = new AdvisorDataSetIterator(dataDirectory, batchSize, false, featureSize, labelSize)
    val buf = scala.collection.mutable.ListBuffer.empty[DataSet]
    while (iterator.hasNext)
      buf += iterator.next
    val rdd = sc.parallelize(buf)

    val tm = new NewParameterAveragingTrainingMaster.Builder(5, 1)
      .averagingFrequency(averagingFrequency)
      .batchSizePerWorker(miniBatchSizePerWorker)
      .saveUpdater(true)
      .workerPrefetchNumBatches(0)
      .repartionData(Repartition.Always)
      .build();

    val trainedNetworkWrapper = new SparkDl4jMultiLayer(sc, network, tm)

    val predictions = rdd.map { x =>
      val labels = x.getLabels.data().asDouble()

      val vector = MLLibUtil.toVector(x.getFeatureMatrix.dup())
      val score = network.output(x.getFeatureMatrix.dup()).data().asDouble()

      items.zip(score)
    }
    predictions
  }

}
