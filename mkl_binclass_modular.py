from shogun.Features import CombinedFeatures, RealFeatures, Labels
from shogun.Kernel import CombinedKernel, GaussianKernel, CustomKernel
from shogun.Classifier import MKLClassification
from numpy import array, where

def load_data ():
	data = []
	labels = []
	with open('diabetes_scale', 'r') as f:
 		for line in f:
			line_list = line.strip().split(' ')
			temp = float(line_list[0])
			line_list = line_list[1:]
			if len(line_list) == 8:
				data += [[float(element.split(':')[1]) for element in line_list]]
				labels += [temp]
	data = array(data)
	labels = array(labels)
	train_data = data[:301,:].T
	train_labels = labels[:301]
	test_data = data[301:,:].T
	test_labels = labels[301:]
	return train_data, test_data, train_labels, test_labels

def mkl_binclass_modular (train_data, testdata, train_labels, test_labels, d1, d2):

        # create some Gaussian train/test matrix
    	tfeats = RealFeatures(train_data)
    	tkernel = GaussianKernel(128, d1)
    	tkernel.init(tfeats, tfeats)
    	K_train = tkernel.get_kernel_matrix()

    	pfeats = RealFeatures(test_data)
    	tkernel.init(tfeats, pfeats)
    	K_test = tkernel.get_kernel_matrix()

    	# create combined train features
    	feats_train = CombinedFeatures()
    	feats_train.append_feature_obj(RealFeatures(train_data))

    	# and corresponding combined kernel
    	kernel = CombinedKernel()
    	kernel.append_kernel(CustomKernel(K_train))
    	kernel.append_kernel(GaussianKernel(128, d2))
    	kernel.init(feats_train, feats_train)

    	# train mkl
    	labels = Labels(train_labels)
    	mkl = MKLClassification()
	
        # not to use svmlight
        mkl.set_interleaved_optimization_enabled(0)

    	# which norm to use for MKL
    	mkl.set_mkl_norm(1) #2,3

    	# set cost (neg, pos)
    	mkl.set_C(1, 1)

    	# set kernel and labels
    	mkl.set_kernel(kernel)
    	mkl.set_labels(labels)

    	# train
    	mkl.train()

    	# test
	# create combined test features
    	feats_pred = CombinedFeatures()
    	feats_pred.append_feature_obj(RealFeatures(test_data))

    	# and corresponding combined kernel
    	kernel = CombinedKernel()
    	kernel.append_kernel(CustomKernel(K_test))
    	kernel.append_kernel(GaussianKernel(128, d2))
    	kernel.init(feats_train, feats_pred)

	# and classify
    	mkl.set_kernel(kernel)
    	output = mkl.apply().get_labels()
	output = [1.0 if i>0 else -1.0 for i in output]
	accu = len(where(output == test_labels)[0]) / float(len(output))
	return accu

if __name__=='__main__':
    	train_data, test_data, train_labels, test_labels = load_data()
    	for d1 in [x * 0.1 for x in range(1, 11)]:
  		for d2 in [x * 0.1 for x in range(1, 11)]:
    			accu = mkl_binclass_modular (train_data, test_data, train_labels, test_labels, d1, d2)
			print '(' + str(d1) + ' ' + str(d2) + '): ' + str(accu)
