#rm nohup.out

#sbt package
#sbt assembly
#nohup 

#hadoop fs -rm -r /user/odia/gdi/current/fm_* &&
#/opt/alti-spark-1.6.1/bin/spark-submit --driver-java-options "-Dlog4j.configuration=file:/home/john/alchemy/log4j.properties" --verbose --master yarn --class com.gale.alchemy.forecast.UserItemFeaturerDriver --deploy-mode client --num-executors 4 --executor-cores 8 --executor-memory 8G --driver-memory 4G --files=src/main/resources/TrendPredictorTest.conf --jars /mnt/ebs0/external-jars/hadoop-lzo-0.4.3.jar /mnt/ebs0/odia/workspace/forecast/target/scala-2.11/spark-deeplearning-assembly-1.0.jar TrendPredictorTest.conf #/home/odia/forecast/target/scala-2.11/spark-deeplearning-assembly-1.0.jar TrendPredictorTest.conf  #&&

/opt/alti-spark-1.6.2/bin/spark-submit --driver-java-options "-Dlog4j.configuration=file:/home/john/alchemy/log4j.properties" --verbose --master yarn --class com.gale.alchemy.forecast.UserItemFeaturerDriver --deploy-mode client --num-executors 4 --executor-cores 8 --executor-memory 8G --driver-memory 4G --files=src/main/resources/TrendPredictorTest.conf --jars /mnt/ebs0/external-jars/hadoop-lzo-0.4.3.jar /mnt/ebs0/odia/workspace/forecast/target/scala-2.11/spark-deeplearning-assembly-1.0.jar TrendPredictorTest.conf #/home/odia/forecast/target/scala-2.11/spark-deeplearning-assembly-1.0.jar TrendPredictorTest.conf

#hadoop fs -getmerge /user/odia/gdi/current/fm_advisors_product_recos_results /mnt/ebs0/odia/fm_advisors_product_recos_results
#&


#hadoop fs -getmerge /user/odia/gdi/current/doos_1 /mnt/ebs0/odia/doos_1 && hadoop fs -getmerge /user/odia/gdi/current/dont_1 /mnt/ebs0/odia/dont_1  
