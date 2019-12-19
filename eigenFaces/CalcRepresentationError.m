function [Total_RMSE, Total_dynamic_range_error] = CalcRepresentationError(DatasetVectors, mean_X, max_X, min_X, W)
 
NumOfTrainingImages = size(DatasetVectors,2);
 
Total_dynamic_range_error = 0;
Total_RMSE = 0;
 
for train_ind = 1:NumOfTrainingImages
    %for each image vector calc the RMSE and the monrlized images RMSE
    xj = DatasetVectors(:,train_ind);
    yj2 = W'*xj;
    xj = xj + mean_X;%original image
    xj_hat = W*yj2 + mean_X;%reconstructed image
    
    %now get the RMSE without normalizing
%    Image_RMSE = mean(abs(xj - xj_hat))/255;    
    Image_Error = abs(xj - xj_hat)/255;    
    Image_RMSE = mean(Image_Error.*Image_Error);    
    Total_RMSE = Total_RMSE + Image_RMSE;
    
    %read the dynamic range of both images
    %origianl image
    xj_min = min(xj);
    xj_max = max(xj);
    %reconstructed image
    xj_hat_min = min(xj_hat);
    xj_hat_max = max(xj_hat);
    
    %normalize both images to the range of [0,1]
    norm_xj = (xj - xj_min) / (xj_max - xj_min);
    norm_xj_hat = (xj_hat - xj_hat_min) / (xj_hat_max - xj_hat_min);
    
    %calc dynamic range error i.e. the RMSE after normalization
%     Image_dynamic_range_error = mean(abs(norm_xj - norm_xj_hat));
%     Total_dynamic_range_error = Total_dynamic_range_error + Image_dynamic_range_error;
    Image_dynamic_range_error = abs(norm_xj - norm_xj_hat);
    Image_dynamic_range_sqr_error = mean(Image_dynamic_range_error .* Image_dynamic_range_error);
    Total_dynamic_range_error = Total_dynamic_range_error + Image_dynamic_range_sqr_error;
    
end
 
% Total_dynamic_range_error = Total_dynamic_range_error / NumOfTrainingImages * 100;
% Total_RMSE = Total_RMSE / NumOfTrainingImages * 100;
 
Total_dynamic_range_error = sqrt(Total_dynamic_range_error / NumOfTrainingImages) * 100;
Total_RMSE = sqrt(Total_RMSE / NumOfTrainingImages) * 100;

