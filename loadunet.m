


newmodelfile = '/rsrch1/ip/dtfuentes/github/kerasimport/debuglog/dscimg/half/adadelta/256/run_a/005020/005/000/tumormodelunet.h5';
newmodelfile = '/rsrch1/ip/dtfuentes/github/kerasimport/debuglog/crossentropy/half/adadelta/256/run_a/005020/005/000/tumormodelunet.h5';
%net = importKerasNetwork(newmodelfile,'OutputLayerType', 'pixelclassification' )
net = importKerasNetwork(newmodelfile,'OutputLayerType', 'regression' )

layers = importKerasLayers(newmodelfile,'ImportWeights', true,'OutputLayerType', 'pixelclassification')
missinglayers = findPlaceholderLayers(layers)


% https://www.mathworks.com/help/vision/ref/semanticseg.html

% evaluate on test image
image = rand(256,256);

[C,scores] = semanticseg(image,layers);

B = labeloverlay(I,C);
figure
imshow(imtile({I,B}))

