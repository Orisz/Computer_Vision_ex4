%% main script Question 1 –Eigen-Faces and PCA for face recognition:
clc;clear all;close all;
oldFolder = cd('eigenFaces');
run eigenface;
cd ..;
%% main script Question 2 –FLD for face recognition:
addpath('eigenFaces');
% load train set
clc;clear all;close all;
run eigenFaces/readYaleFaces;
X = A;
NumOfTrainingImages = size(A,2);
c = 15;

mean_X = mean(double(X),2);
min_X = min(double(X(:)));
max_X = max(double(X(:)));
%%
%comput the fisher basis where A is the train matrix and train_face_id is
%the images labels
[meanvec, basis] = fisherface(A, train_face_id, c);
W = basis;
X = X - repmat(mean_X,1,size(X,2));
%% 2.2 Calculate the representation errors on the train set and test set
%get the representation error for the train images.
[RMSE_Train, Dynamic_range_error_Train] = CalcRepresentationError(X, mean_X, max_X, min_X, W)

yj_train = zeros(c-1,NumOfTrainingImages);
for train_ind = 1:NumOfTrainingImages
    yj_train(:,train_ind) = W'*X(:,train_ind);
end
%get the representation error for the test images.
NumOfImagesInTestSet = 20;
 
 
yj_test = zeros(c-1,NumOfImagesInTestSet);
X_test = zeros(m*n,NumOfImagesInTestSet);
 
for test_ind = 1:NumOfImagesInTestSet
    xj = eval(['image',num2str(test_ind)]);
    xj = single(xj(:));
    X_test(:,test_ind) = xj - mean_X;
    yj_test(:,test_ind) = W'*(xj - mean_X);
end
[RMSE_Test, Dynamic_range_error_Test] = CalcRepresentationError(X_test, mean_X, max_X, min_X, W)

%% Classify the test set
% classify the test images using the ficher basis.
clear person_ids;
for ind = 1:length(train_face_id)
    person_ids{ind} = num2str(train_face_id(ind));
end
person_ids = person_ids.';
 
Mdl = fitcknn(yj_train.',person_ids,'NumNeighbors',4,'Standardize',1);
 
% Find class index in "Classes names":
Classes_indexes = [];
for ind = 1:length(Mdl.ClassNames)
    for ind2 = 1:length(Mdl.ClassNames)
        if ind == str2num(Mdl.ClassNames{ind2})
            Classes_indexes(ind) = ind2;
        end
    end
end
 
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
disp(['The success ratio is: ',num2str(success_ratio,'%.3g'),'%']);
 
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

