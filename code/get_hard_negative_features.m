function hard_neg = get_hard_negative_features(neg_scn_path, w, b, feature_params)

neg_scenes = dir( fullfile( neg_scn_path, '*.jpg' ));

dimen = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
cell_size = feature_params.hog_cell_size;
num_cell = feature_params.template_size/cell_size;

scalar = [1:-0.05:0.85 0.8:-0.1:0.1];
TRUE_THRE = 0.6;

hard_neg = zeros(0, dimen);
for i = 1:length(neg_scenes)
	img = imread( fullfile( neg_scn_path, neg_scenes(i).name ));
	img = single(img)/255;
	if(size(img,3) > 1)
		img = rgb2gray(img);
	end

	for scale = scalar
		% operate on different scale and resize image to the specified scale
		scaled_img = imresize(img, scale);
		HOG = vl_hog(scaled_img, feature_params.hog_cell_size);
		[hog_h, hog_w, ~] = size(HOG);
		cell_row = hog_h - num_cell + 1;
		cell_col = hog_w - num_cell +1;

		% features extracted in the sliding windows
        feature = zeros(cell_row*cell_col,dimen);
		for y = 1:cell_row
			for x = 1:cell_col
				tmp = HOG(y:(y+num_cell-1), x:(x+num_cell-1), :);
				feature((y-1)*cell_col+x,:) = reshape(tmp,1,dimen);
			end
		end
        score = feature*w + b;
		feat = feature(score>TRUE_THRE);
		hard_neg = [hard_neg; feat];

	end
end

end