% Load paths.
if ~isdeployed
  addpath('./nifti');
end

% load pretrained network
newmodelfile = '/rsrch1/ip/dtfuentes/github/kerasimport/debuglog/crossentropy/half/adadelta/256/run_a/005020/005/000/tumormodelunet.h5';
net = importKerasNetwork(newmodelfile,'OutputLayerType', 'pixelclassification' )

% show any missing layers
missinglayers = findPlaceholderLayers(net.Layers)

% https://www.mathworks.com/help/vision/ref/semanticseg.html

% evaluate on test image
filename= 'testdata.nii'
niiimage= load_nii(filename);
image = imresize(niiimage.img(:,:,63),[256,256]);
[C,scores,allScores] = semanticseg(image,net );

% show liver score
figure
imshow(image,[])
figure
liverscore = allScores(17:272,17:272,2);
imshow(liverscore,[])

