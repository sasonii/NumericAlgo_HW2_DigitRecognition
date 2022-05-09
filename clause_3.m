%% ====================== Prepare New Test Set ======================
load('mnist.mat');
load('classifiers.mat');

classifier_count = 28*28+1;

num_images = test.count;
new_test_images = shiftdim(test.images, 2);
A_new_test = reshape(new_test_images,num_images,28*28);
A_new_test = [A_new_test, ones(num_images,1)];
true_labels = test.labels;

%% ============================ Predict ==============================
UNCLASSIFIED = -1;
pred = UNCLASSIFIED * ones(num_images, 1);

histogram = zeros(num_images, 10);
for classifier_index = 0:1:9
    result = sign(A_new_test*classifiers(:,classifier_index + 1));
    classified_indexes = find(result==ones(num_images,1));
    for classified_index=1:1:length(classified_indexes)
        histogram(classified_indexes(classified_index,1), classifier_index+1) = 1;
    end
end

num_of_uncertaincies = 0;

for histogram_row = 1:1:num_images
    ones_counter = 0;
    for histogram_column=1:1:10
        if histogram(histogram_row, histogram_column) == 1
            ones_counter = ones_counter + 1;
            pred(histogram_row,:) = histogram_column - 1;
        end        
    end
    if ones_counter~=1
        num_of_uncertaincies=num_of_uncertaincies+1;
        pred(histogram_row,:) = -1;
    end
end

%% =========================== Evaluate ==============================
acc = mean(pred == true_labels)*100;
disp(['Accuracy=',num2str(acc),'% (',num2str((1-acc/100)*num_images),' wrong examples)']); 
disp(['Number of Unclassified images: ', num2str(num_of_uncertaincies)]);

%% =========================== Fixing Unclassified ===================

% create matrix that contains all possible classifiers between two digits
couple_classifiers = zeros(28*28+1,10,10);
for i= 0:1:9
    for j = 0:1:9
        if i == j
            continue
        end
        if j < i
            couple_classifiers(:,i+1,j+1) = couple_classifiers(:,j+1,i+1); 
        end
        couple_classifiers(:,i+1,j+1) = clause_1_general(i,j);
    end
end
save('couple_classifiers.mat','couple_classifiers');
% images that need better classifaction
error_unclassified = find(pred == UNCLASSIFIED * ones(num_images, 1));
num_of_uncertaincies = 0;
% Classifying the image using elimination each time
for unclassified_index = 1:1:length(error_unclassified)
    histogram_row = error_unclassified(unclassified_index);
    indexes = [-1,-1];
    ones_counter = 0;
    for histogram_column=1:1:10
        if histogram(histogram_row, histogram_column) == 1
            ones_counter = ones_counter + 1;
            indexes(ones_counter) = histogram_column - 1;
        end
        if ones_counter == 2
            if sign(A_new_test(histogram_row,:)*couple_classifiers(:,indexes(1) + 1,indexes(2) + 1)) == 1
                indexes(2) = -1;
                ones_counter = ones_counter -1;
            else
                indexes(1) = indexes(2);
                indexes(2) = -1;
                ones_counter = ones_counter -1;
            end
        end
    end
    if ones_counter == 0
        num_of_uncertaincies = num_of_uncertaincies+1;
        pred(histogram_row,:) = indexes(1);
    end
end

% displaying the better classification
acc = mean(pred == true_labels)*100;
disp(['Accuracy=',num2str(acc),'% (',num2str((1-acc/100)*num_images),' wrong examples)']); 
disp(['Number of Unclassified images: ', num2str(num_of_uncertaincies)]);

error = find(pred~=true_labels); 
    for k=1:1:5 %length(error)
        figure(2);
        imagesc(reshape(A_new_test(error(k),1:28^2),[28,28]));
        colormap(gray(256))
        axis image; axis off; 
        title(['problematic digit number ',num2str(k)]); %,' :',num2str(A_new_test(error(k),:)*x)
        pause;  
    end