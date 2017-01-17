% Starter code prepared by James Hays
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale, because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.
% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);
each = ceil(num_samples/num_images);
dimen = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;

features_neg = zeros(num_images*each,dimen);
for i=1:num_images
	img = rgb2gray(im2single(imread(strcat([non_face_scn_path '/' image_files(i).name]))));
	[height,width] = size(img);
	h_step = floor((height-feature_params.template_size)/each);
	w_step = floor((width-feature_params.template_size)/each);
	for j=0:each
		crop = img(h_step*j+1:h_step*j+feature_params.template_size,...
					w_step*j+1:w_step*j+feature_params.template_size);
		HOG = vl_hog(crop, feature_params.hog_cell_size);	
		features_neg((i-1)*(each+1)+j+1,:) = reshape(HOG,1,dimen);
	end
end

end
