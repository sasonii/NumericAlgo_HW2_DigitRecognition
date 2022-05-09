function x = clause_2_general(digit)
    
 
    %% ======================= Parameters ===========================
    N = 4000;
    digit1 = digit;
    % digit2 = 7;
    
    %%  ==================== Load MNIST dataset ======================
    load('mnist.mat');
    
    % ------- Little bit of exploration to feel the data -------------
%     whos
%     disp(training); 
%     disp(size(training.images));
%     imagesc(training.images(:,:,10)); axis image; colormap(gray(256)); 
%     title(['This image label is ',num2str(training.labels(10))]);
    % ----------------------------------------------------------------
    
    imagesPerDigit1 = training.images(:,:,training.labels == digit1);
    imagesPerDigit2 = training.images(:,:,training.labels ~= digit1);
   
    
    %% ======================= Create A, b ============================
    A_all = zeros(2*N,28^2);
    b_all = zeros(2*N,1);
    for i=1:N
        A_all(2*i-1,:) = reshape(imagesPerDigit1(:,:,i),1,28*28);
        A_all(2*i,:)   = reshape(imagesPerDigit2(:,:,i),1,28*28);
        b_all(2*i-1)   = +1;
        b_all(2*i)     = -1; 
    end
    A_all = [A_all, ones(2*N,1)];
    
    %% ========================= Solve LS ==============================
    A_train = A_all(1:N,:); 
    b_train = b_all(1:N); 
    
    x=pinv(A_train)*b_train; 
    
    %% ===================== Prepare Test Set ==========================
    A_test = A_all(N+1:2*N,:); 
    b_test = b_all(N+1:2*N); 
    
    %% ===================== Check Performance ===========================
    
    predC = sign(A_train*x); 
    trueC = b_train; 
    disp('Train Error:'); 
    acc=mean(predC == trueC)*100;
    disp(['Classifier on digit: ',num2str(digit), '. Accuracy=',num2str(acc),'% (',num2str((1-acc/100)*N),' wrong examples)']); 
    
    predC = sign(A_test*x); 
    trueC = b_test; 
    disp('Test Error:'); 
    acc=mean(predC == trueC)*100;
    disp(['Classifier on digit: ',num2str(digit), '. Accuracy=',num2str(acc),'% (',num2str((1-acc/100)*N),' wrong examples)']); 
    
    %% ================== Show the Problematic Images ====================
    
    error = find(predC~=trueC); 
    for k=1:1:1 %length(error)
        figure(2);
        imagesc(reshape(A_test(error(k),1:28^2),[28,28]));
        colormap(gray(256))
        axis image; axis off; 
        title(['Classifier on digit: ',num2str(digit), '.problematic digit number ',num2str(k),' :',num2str(A_test(error(k),:)*x)]); 
        pause;  
    end

end