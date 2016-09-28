package com.gale.alchemy.forecast

import org.apache.spark.SparkConf
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD

import com.gale.alchemy.forecast.utils.Logs
import com.gale.alchemy.forecast.utils.Logs
import com.gale.alchemy.forecast.utils.SmartStringArray
import com.gale.alchemy.forecast.utils.SmartStringArray
import com.typesafe.config.Config
import com.typesafe.config.ConfigFactory
import com.typesafe.config.ConfigFactory

object ChurnPredictorDriver {
  def main(args: Array[String]): Unit = {
    new ChurnPredictorDriver(args)
  }
}

class ChurnPredictorDriver(args: Array[String]) extends Serializable with Logs {

  info("Starting Churn Prediction Driver Class")

  if (args.length < 1) {
    fatal("ChurnPredictorDriver was called with incorrect arguments: " +
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
      fatal("An error occurred when loading the configuration in the driver. Terminating driver process.", e)
      System.exit(-1)
  }

  var sc = new SparkContext(new SparkConf().setAppName("ChurnPredictorDriver"))

  var instances: RDD[SmartStringArray] = null

  try {
    instances = SmartStringArray.tableFromTextFile("/user/odia/mackenzie/forecast/data", ',', sc)

  } catch {
    case e: Exception =>
      fatal("An error occurred while loading the instances in the driver. Terminating ...", e)
      System.exit(-1)
  }

  val dataset = instances.map { x => (x.toArray.toList.head, x.toArray.toList.tail.mkString(",")) }
    .groupByKey().sortByKey(true)

  val predict = dataset.map(f => (f._1, f._2.filter { x => x.contains("201605") }))
    .map(f => (f._1, f._2.map { x => x.reverse.tail.reverse }))
  val training = dataset.subtract(predict)

  val sqlContext = new org.apache.spark.sql.SQLContext(sc)
  import sqlContext.implicits._
  import org.apache.spark.sql.functions._

  training.map(f => f._2.map { x => (f._1, x) }).flatMap(f => f)
    .toDF("key", "value").write.partitionBy("key").save("/user/odia/mackenzie/forecast/train")

  predict.map(f => f._2.map { x => (f._1, x) }).flatMap(f => f)
    .toDF("key", "value").write.partitionBy("key").save("/user/odia/mackenzie/forecast/test")

  val predictor = new ChurnPredictor(batchSize = 20,
    featureSize = 86,
    nEpochs = 10,
    hiddenLayers = 50,
    miniBatchSizePerWorker = 10,
    averagingFrequency = 5,
    numberOfAveragings = 3,
    learningRate = 0.0018,
    l2Regularization = 1e-5,
    dataDirectory = "/user/odia/mackenzie/forecast/")

  predictor.train()
  val predictions = predictor.predict().zipWithIndex().map(f => (f._2, f._1))
  predictor.sc.stop()

  val z = predict.zipWithIndex().map(f => (f._2, f._1._1)).join(predictions).map(f => f._2)

}