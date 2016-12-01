package com.gale.alchemy.forecast

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import com.gale.alchemy.forecast.utils.Logs
import com.gale.alchemy.forecast.utils.SmartStringArray
import com.typesafe.config.Config
import com.typesafe.config.ConfigFactory
import com.gale.alchemy.conf.Constants

object RecommenderDriver {
  def main(args: Array[String]): Unit = {
    new RecommenderDriver(args)
  }
}

class RecommenderDriver(args: Array[String]) extends Serializable with Logs {

  if (args.length < 1) {
    fatal("RecommenderDriver was called with incorrect arguments: " +
      args.reduce((a, b) => (a + " " + b)))
    fatal("Usage: ChurnPredictorDriver <configFile>")
    System.exit(-1)
  }

  var config: Config = null

  try {
    info("Loading driver configs from: " + args(0))
    config = ConfigFactory.parseResourcesAnySyntax(args(0))
  } catch {
    case e: Exception =>
      fatal("An error occurred when loading the configuration in the driver.", e)
      System.exit(-1)
  }

  val sc = new SparkContext(new SparkConf().setAppName("RecommenderDriver"))

  var items: List[String] = null
  var test: RDD[(String, Long)] = null

  try {
    items = SmartStringArray.tableFromTextFile(Constants.TRAINING, ',', sc)
      .map { x => x(1) }.collect().toList.distinct
    test = SmartStringArray.tableFromTextFile(Constants.TEST, ',', sc)
      .map { x => (x(0), x(0)) }.distinct().sortByKey(true).map(f => f._1).zipWithIndex()
  } catch {
    case e: Exception =>
      fatal("An error occurred while loading the instances in the driver.", e)
      System.exit(-1)
  }

  val predictor = new Recommender(batchSize = 1,
    featureSize = 520, // (64 + 1) * 8
    nEpochs = 5,
    hiddenLayers = 100,
    miniBatchSizePerWorker = 30,
    averagingFrequency = 5,
    numberOfAveragings = 3,
    learningRate = 0.1,
    l2Regularization = 0.001,
    labelSize = 111,
    dataDirectory = Constants.DATA_DIR, sc)

  val net = predictor.train()

  val predictions = predictor.predict(net, items).zipWithIndex().map(f => f._1.map(x => (f._2, x))).flatMap(f => f)
  
  test.map(f => (f._2, f._1)).saveAsTextFile("/user/odia/mackenzie/lstm_recos/testIDs")
  predictions.saveAsTextFile(Constants.RESULTS)
}
