package com.gale.alchemy.forecast;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Stack;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.datavec.api.records.reader.impl.FileRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class StackSequenceRecordReader extends FileRecordReader {

	private Stack<List<Path>> sequenceStack = new Stack<>();
	private FileSystem fs;

	private static final long serialVersionUID = 2949161683514893388L;
	private CSVSequenceRecordReader csvRecordReader = new CSVSequenceRecordReader(); // TODO

	private static final Logger LOG = LoggerFactory.getLogger(StackSequenceRecordReader.class);

	public StackSequenceRecordReader(FileSystem fs) {
		this.fs = fs;
	}

	public void newRecord(Stack<Path> pathStack) {
		List<Path> paths = new ArrayList<>();
		
		if (pathStack.isEmpty()) return;	
		while (!pathStack.isEmpty()) {
			paths.add(pathStack.pop());
		}
		LOG.info("Steps: " + paths.size());
		this.sequenceStack.push(paths);
	}

	public boolean hasNext() {
		return !sequenceStack.isEmpty() && !sequenceStack.peek().isEmpty();
	}

	@Override
	public List<Writable> next() {
		List<Writable> writable = new ArrayList<Writable>();
		List<Path> paths = sequenceStack.pop();
		DataInputStream in = null;
		for (Path p : paths) {
			try {
				in = fs.open(p);
				List<List<Writable>> steps = csvRecordReader.sequenceRecord(p.toUri(), in);
				if (!CollectionUtils.isEmpty(steps)) {
					for (List<Writable> instance : steps) {
						writable.addAll(instance);
					}
				}
			} catch (IOException e) {
				e.printStackTrace();
			} finally {
				if (in != null) {
					IOUtils.closeQuietly(in);
				}
			}
		}
		return writable;
	}

	private List<List<Writable>> records() {
		List<List<Writable>> writables = new ArrayList<>();
		while (hasNext()) {
			writables.add(next());
		}
		return writables;
	}

	public MultiDataSet toMultiDataSet(int numFeatures, int numLabels) {
		List<List<Writable>> records = records();

		INDArray[] inputArrs = new INDArray[records.size()];
		INDArray[] outputArrs = new INDArray[records.size()];

		List<MultiDataSet> list = new ArrayList<>(records.size());

		for (int i = 0; i < records.size(); i++) {
			Pair<INDArray, INDArray> pair = 
					toPair(records.get(i), numFeatures, numLabels);
			list.add(new org.nd4j.linalg.dataset.MultiDataSet(pair.getFirst(), pair.getSecond()));
		}
		return org.nd4j.linalg.dataset.MultiDataSet.merge(list);
	}

	private Pair<INDArray, INDArray> toPair(Collection<Writable> record, int numFeatures, int numLabels) {
		Iterator<Writable> writables = record.iterator();
		Writable firstWritable = writables.next();
		INDArray vector1 = Nd4j.create(numFeatures);
		INDArray vector2 = Nd4j.create(numFeatures);//(numLabels);

		vector1.putScalar(0, firstWritable.toDouble());

		int count1 = 1, count2 = 0, count = 0;
		while (writables.hasNext()) {
			Writable w = writables.next();
			if (count1 < numFeatures) {
				vector1.putScalar(count1++, w.toDouble());
			} else {
				if (count2 < numLabels)
					vector2.putScalar(count2++, w.toDouble());
			}
			count++;
		}
		return new Pair<>(vector1, vector2);

	}
}

