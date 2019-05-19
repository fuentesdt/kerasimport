

modelfile = '/rsrch1/ip/dtfuentes/github/livermask/ModelZoo/livermask/tumormodelunet.json';
weightfile = '/rsrch1/ip/dtfuentes/github/livermask/ModelZoo/livermask/tumormodelunet.h5';
net = importKerasNetwork(modelfile,'WeightFile',weightfile,'OutputLayerType', 'classification' )

newmodelfile = './debuglog/dscimg/half/adadelta/256/run_a/005020/005/000/tumormodelunet.h5';
layers = importKerasLayers(newmodelfile,'ImportWeights', true )
missinglayers = findPlaceholderLayers(layers)
