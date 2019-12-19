% load train set (1.a)
readYaleFaces;
 
% A - is the training set matrix where each column is a face image
% train_face_id - an array with the id of the faces of the training set.
% image1--image20 are the test set.
% is_face - is an array with 1 for test images that contain a face
% faec_id - is an array with the id of the face in the test set, 
%           0 if no face and -1 if a face not from the train-set.
 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Your Code Here  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%% Subtract mean image (1.b)
 
X = A;
 
mean_X = mean(double(X),2);
figure; 
imagesc(reshape(mean_X,m,n));
colormap('jet');
colorbar;
title('Mean face of training set');
 
% mean_X = mean(double(X(:)));
 
min_X = min(double(X(:)));
max_X = max(double(X(:)));
X = X - repmat(mean_X,[1, size(X,2)]);
 
 
% Compute eigenvectors and report first 5 eigen-faces (2)
 
[U5,S5,V5] = svds(X,5);
 
figure;
for ind = 1:5
    subplot(3,2,ind);
    im_to_show = reshape(U5(:,ind),m,n);
%     max_im_to_show = max(abs(im_to_show(:)));
%     im_to_show = uint8(round((255/max_im_to_show).*im_to_show));
    imagesc(im_to_show);
    title(['Eigen-face number ',num2str(ind),', with eigen-value: ',num2str(S5(ind,ind))]);
    colormap('jet')
    colorbar;
end
 
 
 
%% Display and compute the representation error for the training images (3)
 
NumOfTrainingImages = size(A,2);
NumOfEigenVectors = min(25,size(A,2));
 
[W,S25,~] = svds(X,NumOfEigenVectors);
 
yj_train = zeros(NumOfEigenVectors,NumOfTrainingImages);
for train_ind = 1:NumOfTrainingImages
    yj_train(:,train_ind) = W'*X(:,train_ind);
end
 
figure;
face_id_found = zeros(15,1);
for ind = 1:NumOfTrainingImages
    if face_id_found(train_face_id(ind))
        continue
    end
    face_id_found(train_face_id(ind)) = 1;
    
    subplot(3,5,train_face_id(ind));
    im_to_show = reshape(A(:,ind),m,n);
    max_im_to_show = max(abs(im_to_show(:)));
%     im_to_show = uint8(round((255/max_im_to_show).*im_to_show));
    imagesc(im_to_show);
    colormap('gray');
    caxis([0 170])
    title(['FaceID: ',num2str(train_face_id(ind))]);
 
end
 
figure;
face_id_found = zeros(15,1);
for ind = 1:NumOfTrainingImages
    if face_id_found(train_face_id(ind))
        continue
    end
    face_id_found(train_face_id(ind)) = 1;
    
    xj = X(:,ind);
    yj = W'*xj;
    subplot(3,5,train_face_id(ind));
%     max_im_to_show = max(abs(im_to_show(:)));
%     im_to_show = uint8(round((255/max_im_to_show).*im_to_show));
    im_to_show = reshape(mean_X  + W*yj,m,n);
    imagesc(im_to_show)
    colormap('gray')
    title(['FaceID: ',num2str(train_face_id(ind))]);
    caxis([0 170])
 
end
 
 
[RMSE_Train, Dynamic_range_error_Train] = CalcRepresentationError(X, mean_X, max_X, min_X, W)
 
%% Compute the representation error for the test images. Classify the test images and report error rate (4)
 
NumOfImagesInTestSet = 20;
 
[W,S25,~] = svds(X,NumOfEigenVectors);
 
yj_test = zeros(NumOfEigenVectors,NumOfImagesInTestSet);
X_test = zeros(m*n,NumOfImagesInTestSet);
 
for test_ind = 1:NumOfImagesInTestSet
    xj = eval(['image',num2str(test_ind)]);
    xj = single(xj(:)) - mean_X;
    X_test(:,test_ind) = xj;
    yj_test(:,test_ind) = W'*xj;
    
end
 
[RMSE_Test, Dynamic_range_error_Test] = CalcRepresentationError(X_test, mean_X, max_X, min_X, W)
 
clear person_ids;
for ind = 1:length(train_face_id)
    person_ids{ind} = num2str(train_face_id(ind));
end
person_ids = person_ids.';
 
Mdl = fitcknn(yj_train.',person_ids,'NumNeighbors',3,'Standardize',1);
 
% Find class index in "Classes names":
Classes_indexes = [];
for ind = 1:length(Mdl.ClassNames)
    for ind2 = 1:length(Mdl.ClassNames)
        if ind == str2num(Mdl.ClassNames{ind2})
            Classes_indexes(ind) = ind2;
        end
    end
end
 
% Success ratio for the training set:
[label,score,cost] = predict(Mdl,yj_train.');
 
success_ratio = 0;
number_of_real_faces = 0;
for train_ind = 1:NumOfTrainingImages
    number_of_real_faces = number_of_real_faces+1;
    if str2num(label{train_ind}) == train_face_id(train_ind)
        success_ratio = success_ratio+1;
    end
end
success_ratio = success_ratio/number_of_real_faces*100;
disp(['The success ratio for the train set is: ',num2str(success_ratio,'%.3g'),'%']);
 
% Success ratio for the test set:
[label,score,cost] = predict(Mdl,yj_test.');
 
success_ratio = 0;
number_of_real_faces = 0;
for test_ind = 1:NumOfImagesInTestSet
    if (face_id(test_ind) > 0)
        number_of_real_faces = number_of_real_faces+1;
        if str2num(label{test_ind}) == face_id(test_ind)
            success_ratio = success_ratio+1;
        end
    end
end
success_ratio = success_ratio/number_of_real_faces*100;
disp(['The success ratio for the test set is: ',num2str(success_ratio,'%.3g'),'%']);
 
 
 
figure;
for test_ind = 1:NumOfImagesInTestSet
    xj = eval(['image',num2str(test_ind)]);
    subplot(5,4,test_ind);
    im_to_show = xj;
    imagesc(im_to_show)
    colormap('gray')
%     colorbar;
    title(['FaceID: ',num2str(face_id(test_ind))]);
end
 
figure;
for test_ind = 1:20
    subplot(5,4,test_ind);
    im_to_show = reshape(mean_X + W*yj_test(:,test_ind),m,n);
    imagesc(im_to_show)
    colormap('gray')
%     colorbar;
    caxis([0 255]);
    title(['FaceID: ',label{test_ind},'   Score: ',num2str(score(test_ind,Classes_indexes(str2num(label{test_ind}))))]);
 
end
