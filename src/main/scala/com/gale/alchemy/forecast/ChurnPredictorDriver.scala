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

  val dataset = instances.map { x => (x.toArray.toList.head, x.toArray.toList.tail) }.groupByKey().sortByKey(true)

  val labeledPoints = dataset.map { f =>
    val instances: List[List[String]] = List()
    (0 until f._2.toList.size - 1).foreach { i =>
      instances +: (f._2.toList +: f._2.toList(i + 1).reverse.head)
    }
    (f._1, instances.toIterable)
  }

  val predict = dataset.map(f => (f._1, f._2.map { x => x.filter { date => date == "201605" } }))
    .map(f => (f._1, f._2.map { x => x.reverse.tail.reverse }))
  val training = dataset.subtract(predict)

  training.sortByKey(true).map { x =>
    sc.parallelize(x._2.toList.reverse.tail.reverse).repartition(1) // coalesce(1,true)
      .saveAsTextFile("/user/odia/mackenzie/forecast/train/" + x._1)
  }
  predict.sortByKey(true).map { x =>
    sc.parallelize(x._2.toList.reverse.tail.reverse).repartition(1) // coalesce(1,true)
      .saveAsTextFile("/user/odia/mackenzie/forecast/test/" + x._1)
  }

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