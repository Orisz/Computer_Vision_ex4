function [meanvec, basis] = fisherface(face_data, label_train, c)
% FISHERFACE Creates the Fisher face basis for a set of images
%
% face_data - cell array containing the face images 
% label_train - the training labels matching c classes
% c - number of classes to use
%
% meanvec - mean vector of the face images
% basis - Fisher basis of the face images
%


%face_data - data matrix where the num of rows is the problem dim and num
%of cols is the num of examples.
% a) find the basis rank, 
N = size(face_data,2);
r = N - c;

%find the basis using pca method and svd for run time optimization
X = face_data;
mean_X = mean(double(X),2);
meanvec = mean_X;
% min_X = min(double(X(:)));
% max_X = max(double(X(:)));
X = X - repmat(mean_X,1,size(X,2));
% Compute most dominant r eigenvectors 
[Ur,~,~] = svds(X,r);

%this is the required W pca of 'r' basis
Wpca = Ur;

% b) project the train images in Wpca
yj_train = zeros(r,N);
for train_ind = 1:N
    yj_train(:,train_ind) = Wpca'*X(:,train_ind);
end

% c) calc Sw and Sb
Sw = zeros(r);%an r by r matrix
Sb = zeros(r);%an r by r matrix
classes_means = zeros(r,c);
classes_size = zeros(1,c);
for k = 1:c
   idxs = find(label_train == k);
   Nk = length(idxs);
   class_k_train = yj_train(:,idxs);
   classes_means(:,k) = mean(class_k_train,2);
   class_k_train_centered = class_k_train - repmat(classes_means(:,k),1,size(class_k_train,2));
   Sw = Sw + class_k_train_centered * class_k_train_centered';
   classes_size(k) = Nk;
end
total_mean = mean(classes_means,2);
means_centered = classes_means -  repmat(total_mean,1,size(classes_means,2));
for i = 1:c
   Sb = Sb + classes_size(i)*(means_centered(:,i) * means_centered(:,i)');
end

% d) solve the generelized eigen vector problem
[Wfld,~] = eigs(Sb, Sw, c - 1);
basis = Wpca * Wfld;
end

