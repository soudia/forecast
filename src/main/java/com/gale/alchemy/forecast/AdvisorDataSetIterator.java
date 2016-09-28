package com.gale.alchemy.forecast;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

public class AdvisorDataSetIterator implements DataSetIterator {

	private static final long serialVersionUID = -8938692790057042187L;

	private int cursor = 0;
	private int vectorSize = 0;
	private final int batchSize;

	private volatile FileSystem fs;
	private volatile String hdfsUrl;
	protected volatile RemoteIterator<LocatedFileStatus> hdfsIterator;

	private static final String HDFS_URL  = "hdfs://nn-galepartners.s3s.altiscale.com:8020";
	private static final String DATA_DIR  = "/user/odia/mackenzie/forecast/";
	private static final String CORE_SITE = "/etc/hadoop/conf/core-site.xml";
	private static final String HDFS_SITE = "/etc/hadoop/conf/hdfs-site.xml";

	public AdvisorDataSetIterator(int batchSize, boolean train) {
		this(DATA_DIR, batchSize, train);
	}

	public AdvisorDataSetIterator(String dataDirectory, int batchSize, boolean train) {
		this.batchSize = batchSize;
		int pos = dataDirectory.lastIndexOf("/");
		dataDirectory = (pos > -1 ? dataDirectory.substring(0, pos) : dataDirectory);
		this.hdfsUrl = HDFS_URL + dataDirectory + (train ? "train/" : "test/");
		initialize();
	}

	private void initialize() {
		Configuration configuration = new Configuration();
		configuration.addResource(new Path(CORE_SITE));
		configuration.addResource(new Path(HDFS_SITE));

		configuration.set("fs.hdfs.impl", org.apache.hadoop.hdfs.DistributedFileSystem.class.getName());
		configuration.set("fs.file.impl", org.apache.hadoop.fs.LocalFileSystem.class.getName());

		try {
			fs = FileSystem.get(configuration);
			hdfsIterator = fs.listFiles(new Path(hdfsUrl), true);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public boolean hasNext() {
		try {
			return hdfsIterator != null && hdfsIterator.hasNext();
		} catch (IOException e) {
			return false;
		}
	}

	@Override
	public DataSet next() {
		return next(batchSize);
	}

	@Override
	public boolean asyncSupported() {
		return false;
	}

	@Override
	public int batch() {
		return batchSize;
	}

	@Override
	public int cursor() {
		return cursor;
	}

	@Override
	public List<String> getLabels() { // TODO
		return null;
	}

	@Override
	public DataSetPreProcessor getPreProcessor() {
		return null;
	}

	@Override
	public int inputColumns() { // TODO
		return 0;
	}

	@Override
	public DataSet next(int num) {
		try {
			if (!hdfsIterator.hasNext())
				throw new NoSuchElementException();
			return nextDataSet(num);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	private DataSet nextDataSet(int num) throws IOException {
		List<List<Double>> instances = new ArrayList<List<Double>>(num);
		List<Double> targets = new ArrayList<Double>();

		for (int i = 0; i < num && hdfsIterator.hasNext(); i++) {

			LocatedFileStatus next = hdfsIterator.next();
			Path path = next.getPath();

			Reader reader = new InputStreamReader(fs.open(path));
			LineIterator iter = IOUtils.lineIterator(reader);
			while (iter.hasNext()) {
				String[] tokens = iter.next().split(",");
				List<Double> features = new ArrayList<Double>();
				int j = 0;
				for (; j < tokens.length - 1; j++) {
					features.add(Double.valueOf(tokens[j].trim()));
				}
				instances.add(features);
				targets.add(Double.valueOf(tokens[j].trim()));
			}
			cursor++;
		}

		vectorSize = instances.get(0).size(); // number of features
		INDArray labels = Nd4j.create(instances.size(), 1, vectorSize); // one
																		// class
		INDArray features = Nd4j.create(instances.size(), vectorSize, vectorSize);

		for (int i = 0; i < instances.size(); i++) {
			List<Double> instance = instances.get(i);
			INDArray array = Nd4j.create(instance.size());
			for (int j = 0; j < instance.size(); j++) {
				array.putScalar(j, instance.get(j));
			}
			features.put(i, array);
			labels.putScalar(i, targets.get(i));
		}

		return new DataSet(features, labels);
	}

	@Override
	public int numExamples() { // TODO
		return 0;
	}

	@Override
	public void reset() {
		try {
			hdfsIterator = fs.listFiles(new Path(hdfsUrl), true);
		} catch (IllegalArgumentException | IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public boolean resetSupported() { // TODO
		return false;
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor processor) {
	}

	@Override
	public int totalExamples() { // TODO
		return 0;
	}

	@Override
	public int totalOutcomes() { // TODO
		return 1;
	}
	
	public void remove() {
		
	}

}
