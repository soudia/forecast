#rm nohup.out

#sbt package
#sbt assembly
#nohup 

#hadoop fs -rm -r /user/odia/mackenzie/dl4j/* &&
/opt/alti-spark-2.0.1/bin/spark-submit --driver-java-options "-Dlog4j.configuration=file:/home/john/alchemy/log4j.properties" --verbose --master yarn --class com.gale.alchemy.forecast.RecommenderDriver --deploy-mode client --num-executors 15 --executor-cores 5 --executor-memory 19G --driver-memory 17G --files=src/main/resources/TrendPredictorTest.conf --jars /mnt/ebs0/external-jars/hadoop-lzo-0.4.3.jar /mnt/ebs0/odia/workspace/forecast/target/scala-2.11/spark-deeplearning-assembly-1.0.jar TrendPredictorTest.conf &&
hadoop fs -rm -r /user/odia/mackenzie/dl4j/*

# --class my.class.name.here --num-executors 4 --executor-cores 8 --executor-memory 4G --driver-memory 4G --conf "spark.executor.extraJavaOptions=-Dorg.bytedeco.javacpp.maxbytes=5368709120" --conf "spark.driver.extraJavaOptions=-Dorg.bytedeco.javacpp.maxbytes=5368709120" --conf spark.yarn.executor.memoryOverhead=6144
# https://goo.gl/PlSyGU
