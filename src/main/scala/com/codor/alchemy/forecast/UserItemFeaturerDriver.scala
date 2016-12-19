package codor.alchemy.drivers

import org.apache.spark.HashPartitioner
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToOrderedRDDFunctions
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions

import com.codor.alchemy.conf.Constants
import com.codor.alchemy.forecast.utils.Logs
import com.codor.alchemy.forecast.utils.RDDMultipleTextOutputFormat
import com.codor.alchemy.forecast.utils.SmartStringArray
import com.typesafe.config.Config
import com.typesafe.config.ConfigFactory

object UserItemFeaturerDriver {
  def main(args: Array[String]): Unit = {
    new UserItemFeaturerDriver(args)
  }
}

class UserItemFeaturerDriver(args: Array[String]) extends Serializable with Logs {

  info("Starting UserItemFeaturer Driver Class")

  if (args.length < 1) {
    fatal("UserItemFeaturer was called with incorrect arguments: " +
      args.reduce((a, b) => (a + " " + b)))
    fatal("Usage: UserItemFeaturer <configFile>")
    System.exit(-1)
  }

  var config: Config = null

  try {
    info("Loading driver configs from: " + args(0))
    config = ConfigFactory.parseResourcesAnySyntax(args(0))
  } catch {
    case e: Exception =>
      fatal("An error occurred when loading the configuration in the driver. Exiting", e)
      System.exit(-1)
  }

  val sc = new SparkContext(new SparkConf().setAppName("UserItemFeaturer"))

  var transactions: RDD[SmartStringArray] = null
  var factors: RDD[Array[Double]] = null
  var correlations: RDD[(String, String, Double)] = null

  try {
    transactions = SmartStringArray.tableFromTextFile(Constants.TRANSACTIONS, ',', sc)

    factors = SmartStringArray.tableFromTextFile(Constants.FM_FACTORS, ',',
      sc).map { x => x.toArray.map { x => x.toDouble } }

    correlations = SmartStringArray.tableFromTextFile(Constants.CORRELATIONS, ',',
      sc).map { x => (x(0).replace("(", ""), x(1), x(3).toDouble) }
      .map { f => (f._1, f._2, math.log(f._3 / (f._3 - 1))) }
  } catch {
    case e: Exception =>
      fatal("An error occurred while loading the transactions in the driver. ", e)
      System.exit(-1)
  }

  var nonulls = transactions.filter { x => x(0) != "NULL" && x(1) != "NULL" }

  val fmTrainingData = SmartStringArray.tableFromTextFile(
    "/user/myhome/myfolder/product_recos/training_data_fm_201604_201608_en_purged", ',', sc)
  val users = fmTrainingData.map { x => x(0) }.distinct().collect().sortWith((f1, f2) => f1 < f2).toList
  val items = fmTrainingData.map { x => x(1) }.distinct().collect().sortWith((f1, f2) => f1 < f2).toList

  //val items = nonulls.map { x => x(1) }.distinct().collect().sortWith((f1, f2) => f1 < f2).toList
  //val users = nonulls.map { x => x(0) }.distinct().collect().sortWith((f1, f2) => f1 < f2).toList

  val features = featurize(nonulls, users)

  val indexedFactors = factors.zipWithIndex().map(f => (f._2, f._1))

  //getEmbeddings(features, indexedFactors, users, items)

  getEmbeddings(features, correlations, indexedFactors, users, items)

  /**
   * Featurizes the input data
   */
  def featurize(training: RDD[SmartStringArray], users: List[String]) = {

    val fmUsers = fmTrainingData.map { x => x(0) }.distinct().collect().toList
    val fmItems = fmTrainingData.map { x => x(1) }.distinct().collect().toList
    val nonulls = training.filter { x => fmUsers.contains(x(0)) && fmItems.contains(x(1)) }

    val riskScoreIndex = 2
    val riskScore = nonulls.map { t =>
      if (t(riskScoreIndex) == "NULL") -1 else
        t(riskScoreIndex).toDouble
    }.countByValue()

    val variables = List(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) // TODO: conf

    // RDD[((String, String), Iterable[(String, List[(Int, Double)])])]
    val tuples = nonulls.map { x => // keys: user, year_month
      ((x(0), x(13)), (x(1), variables.map { f =>
        (f, if (x(f) == "NULL") -1 else x(f).toDouble)
      }))
    }.groupByKey()

    // RDD[((String, String), Iterable[List[(Int, Double)]])]
    val discretized = tuples.map { f =>
      (f._1, f._2.map { x =>
        val index = x._2.map(t => t._1).indexOf(riskScoreIndex)
        val old = x._2(index)
        (x._1, x._2.updated(index, (riskScoreIndex,
          riskScore.getOrElse(old._2, 0L).toDouble)))
      })
    }.sortByKey(true)

    val normalized = discretized.map { x => // discretized tuples
      val sumap = x._2.map { t => t._2 }.flatMap(f => f).groupBy(f => f._1)
        .toMap.map(f => (f._1, f._2.map(t => t._2)))

      (x._1, x._2.map { t =>
        (t._1, t._2.map { a => math.abs(a._2) / sumap.getOrElse(a._1, List(1.0)).sum
        })
      })
    }

    //user index, (user, Iterable[(item, yearmon, Double)])
    discretized.map { t => (t._1, t._2.map(f => f._1)) }.join(normalized)
      .map(f => (f._1._1, f._2._2.map(t => (t._1, f._1._2, t._2))))
      .map(f => (users.indexOf(f._1).toLong, f))
  }

  /**
   * Generates the embeddings for every (advisor, yearmon) combination.
   */
  def getEmbeddings(features: RDD[(Long, (String, Iterable[(String, String, List[Double])]))],
    factors: RDD[(Long, Array[Double])], users: List[String], items: List[String]) = {

    val itemFactors = factors.filter(f => f._1 >= users.size && f._1 < users.size +
      items.size).map(f => (f._1, f._2)).collect().map(f => f._2)
    val size = itemFactors(0).size
    val flattened = itemFactors.flatMap { x => x }

    info("Flattened:" + features.first()._2._1 + ", " + features.first()._2._2.toList)

    //user index, (user, Iterable[(item, yearmon, Double)], fmFactor)
    val userFactors = features.join(factors).map(f => (f._2._1._1, f._2._1._2, f._2._2))

    val embeddings = userFactors.map { t =>
      var label: List[Double] = List.fill(items.size)(0.0)

      val map = t._2.map(f => (f._2, f._1)).groupBy(f => f._1).toMap
        .map(f => (f._1, f._2.map(f => f._2)))

      t._2.groupBy(f => f._2).map { c =>
        var embedding = List.fill((64 + 1) * size)(0.0) // (items.size + 1) * size
        (0 until t._3.size).foreach { i => embedding = embedding.updated(i, t._3(i)) }
        var net = 0.0
        map.getOrElse(c._1, List()).foreach { x => label = label.updated(items.indexOf(x), 1.0) }
        var counter = size
        c._2.map { u =>
          val index = items.indexOf(u._1)
          (counter until (counter + size)).foreach { i =>
            // Uncomment the following to create an embedding of size (items.size + 1) * size
            //((size * (1 + index)) until (size * (1 + index) + size)).foreach { i =>
            embedding = embedding.updated(i, flattened(i - size))
          }
          counter += size
          net += u._3(u._3.size - 1) / c._2.size
        }
        ((t._1, c._1.toInt), (embedding :+ net).mkString(",") + "," + label.mkString(","))
      }
    }.flatMap(f => f).map { f =>
      if (f._1._2 < 201609) {
        ("train/" + f._1._1 + "_" + f._1._2, f._2)
      } else {
        ("test/" + f._1._1 + "_" + f._1._2, f._2)
      }
    }

    embeddings.partitionBy(new HashPartitioner(5000))
      .saveAsHadoopFile("/user/myhome/myfolder/lstm_recos/", classOf[String], classOf[String],
        classOf[RDDMultipleTextOutputFormat])
  }

  def getEmbeddings(features: RDD[(Long, (String, Iterable[(String, String, List[Double])]))],
    userItemCorr: RDD[(String, String, Double)], factors: RDD[(Long, Array[Double])],
    users: List[String], items: List[String]) = {

    val itemFactors = factors.filter(f => f._1 >= users.size && f._1 < users.size +
      items.size).map(f => (f._1, f._2)).collect().map(f => f._2)
    val size = itemFactors(0).size
    val flattened = itemFactors.flatMap { x => x }

    info("Flattened: " + flattened.size + " - " + users.size + " - " + items.size)

    val correlations = userItemCorr.map(f => (f._1, (f._2, f._3))).groupByKey().sortByKey()
      .zipWithIndex().map(f => (f._2, f._1._2)).map(f => (f._1,
        f._2.map(x => (x._1, math.exp(x._2) / f._2.map(f => math.exp(f._2)).sum))))

    val userFeatures = features.join(factors).join(correlations).map(f => (f._2._1._1._1,
      f._2._1._1._2, f._2._1._2, f._2._2))

    val embeddings = userFeatures.map { t =>
      var label: List[Double] = List.fill(items.size)(0.0)

      val map = t._2.map(f => (f._2, f._1)).groupBy(f => f._1).toMap
        .map(f => (f._1, f._2.map(f => f._2)))
      val correlations = t._4.toMap

      t._2.groupBy(f => f._2).map { c =>
        var embedding = List.fill(73 * size)(0.0) // (items.size + 1) * size
        var nets = List.fill(73)(0.0)
        map.getOrElse(c._1, List()).foreach { x => label = label.updated(items.indexOf(x), 1.0) }
        var position = 0
        var updateAt = 0
        c._2.map { u =>
          val index = items.indexOf(u._1)
          ((size * index) until ((1 + index) * size)).foreach { i =>
            embedding = embedding.updated(position, flattened(i))
            position = position + 1
          }
          nets = nets.updated(updateAt, u._3(u._3.size - 1) / c._2.size)
          updateAt = updateAt + 1
        }
        ((t._1, c._1.toInt), (embedding ::: nets).mkString(",") + "," + label.mkString(","))
      }
    }.flatMap(f => f).map { f =>
      if (f._1._2 < 201609) {
        ("train/" + f._1._1 + "_" + f._1._2, f._2)
      } else {
        ("test/" + f._1._1 + "_" + f._1._2, f._2)
      }
    }

    embeddings.partitionBy(new HashPartitioner(5000))
      .saveAsHadoopFile("/user/myhome/myfolder/lstm_recos/", classOf[String], classOf[String],
        classOf[RDDMultipleTextOutputFormat])
  }

}
