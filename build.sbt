name := "spark-deeplearning"

version := "1.0"

scalaVersion := "2.11.8"
resolvers += Resolver.mavenLocal
libraryDependencies += "org.deeplearning4j" % "dl4j-spark-nlp_2.11" % "0.6.0"
libraryDependencies += "org.nd4j" % "nd4j-native" % "0.6.0"
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.0.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.0.0"
libraryDependencies += "com.typesafe" % "config" % "1.2.1"
libraryDependencies += "org.scalatest" % "scalatest_2.10" % "2.0" % "test" 
libraryDependencies += "junit" % "junit" % "4.10"  //adding junit



//libraryDependencies += "org.deeplearning4j" % "dl4j-spark" % "0.0.3.3.3.alpha1-SNAPSHOT"
