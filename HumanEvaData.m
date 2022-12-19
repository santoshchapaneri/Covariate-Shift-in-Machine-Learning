clear; clc;
% load HoG features
load('HoG_All_TrainValidation_C1C2C3.mat');

%% Collect features and poses for each subject
hogS1 = []; hogS2 = []; hogS3 = [];
poseS1 = []; poseS2 = []; poseS3 = [];
C1 = 1:270; C2 = 271:540; C3 = 541:810;

indexS1 = [];
indexS1 = [indexS1 ; find(motionframe(:,1) == 1)];
indexS1 = [indexS1 ; find(motionframe(:,1) == 2)];
indexS1 = [indexS1 ; find(motionframe(:,1) == 3)];
indexS1 = [indexS1 ; find(motionframe(:,1) == 4)];
indexS1 = [indexS1 ; find(motionframe(:,1) == 5)];
hogS1 = hog(indexS1,:);
poseS1 = pose(indexS1,:);

indexS2 = [];
indexS2 = [indexS2 ; find(motionframe(:,1) == 6)];
indexS2 = [indexS2 ; find(motionframe(:,1) == 7)];
indexS2 = [indexS2 ; find(motionframe(:,1) == 8)];
indexS2 = [indexS2 ; find(motionframe(:,1) == 9)];
indexS2 = [indexS2 ; find(motionframe(:,1) == 10)];
hogS2 = hog(indexS2,:);
poseS2 = pose(indexS2,:);

indexS3 = [];
indexS3 = [indexS3 ; find(motionframe(:,1) == 11)];
indexS3 = [indexS3 ; find(motionframe(:,1) == 12)];
indexS3 = [indexS3 ; find(motionframe(:,1) == 13)];
indexS3 = [indexS3 ; find(motionframe(:,1) == 14)];
indexS3 = [indexS3 ; find(motionframe(:,1) == 15)];
hogS3 = hog(indexS3,:);
poseS3 = pose(indexS3,:);

%% Scenario A
hogA_train = [hogS1(:,C1);hogS2(:,C1);hogS3(:,C1)];
hogA_test = hogS1(:,C1);
poseA_train = [poseS1;poseS2;poseS3];
poseA_test = poseS1;

%% Scenario B
hogB_train = [hogS1(:,C1);hogS2(:,C1);hogS3(:,C1)];
hogB_test = hogS2(:,C1);
poseB_train = [poseS1;poseS2;poseS3];
poseB_test = poseS2;

%% Scenario C
hogC_train = [hogS1(:,C1);hogS2(:,C1);hogS3(:,C1)];
hogC_test = hogS3(:,C1);
poseC_train = [poseS1;poseS2;poseS3];
poseC_test = poseS3;

%% Scenario D
hogD_train = [hogS1;hogS2;hogS3];
hogD_test = hogS1;
poseD_train = [poseS1;poseS2;poseS3];
poseD_test = poseS1;

%% Scenario E
hogE_train = [hogS1;hogS2;hogS3];
hogE_test = hogS2;
poseE_train = [poseS1;poseS2;poseS3];
poseE_test = poseS2;

%% Scenario F
hogF_train = [hogS1;hogS2;hogS3];
hogF_test = hogS3;
poseF_train = [poseS1;poseS2;poseS3];
poseF_test = poseS3;

%% Scenario G
hogG_train = [hogS2(:,C1);hogS3(:,C1)];
hogG_test = hogS1(:,C1);
poseG_train = [poseS2;poseS3];
poseG_test = poseS1;

%% Scenario H
hogH_train = [hogS1(:,C1);hogS3(:,C1)];
hogH_test = hogS2(:,C1);
poseH_train = [poseS1;poseS3];
poseH_test = poseS2;

%% Scenario I
hogI_train = [hogS1(:,C1);hogS2(:,C1)];
hogI_test = hogS3(:,C1);
poseI_train = [poseS1;poseS2];
poseI_test = poseS3;

%% Put in one structure
HumanEvaDataset.hogS1 = hogS1;
HumanEvaDataset.hogS2 = hogS2;
HumanEvaDataset.hogS3 = hogS3;
HumanEvaDataset.poseS1 = poseS1;
HumanEvaDataset.poseS2 = poseS2;
HumanEvaDataset.poseS3 = poseS3;
HumanEvaDataset.hogA_train = hogA_train;
HumanEvaDataset.hogB_train = hogB_train;
HumanEvaDataset.hogC_train = hogC_train;
HumanEvaDataset.hogD_train = hogD_train;
HumanEvaDataset.hogE_train = hogE_train;
HumanEvaDataset.hogF_train = hogF_train;
HumanEvaDataset.hogG_train = hogG_train;
HumanEvaDataset.hogH_train = hogH_train;
HumanEvaDataset.hogI_train = hogI_train;
HumanEvaDataset.hogA_test = hogA_test;
HumanEvaDataset.hogB_test = hogB_test;
HumanEvaDataset.hogC_test = hogC_test;
HumanEvaDataset.hogD_test = hogD_test;
HumanEvaDataset.hogE_test = hogE_test;
HumanEvaDataset.hogF_test = hogF_test;
HumanEvaDataset.hogG_test = hogG_test;
HumanEvaDataset.hogH_test = hogH_test;
HumanEvaDataset.hogI_test = hogI_test;
HumanEvaDataset.poseA_train = poseA_train;
HumanEvaDataset.poseB_train = poseB_train;
HumanEvaDataset.poseC_train = poseC_train;
HumanEvaDataset.poseD_train = poseD_train;
HumanEvaDataset.poseE_train = poseE_train;
HumanEvaDataset.poseF_train = poseF_train;
HumanEvaDataset.poseG_train = poseG_train;
HumanEvaDataset.poseH_train = poseH_train;
HumanEvaDataset.poseI_train = poseI_train;
HumanEvaDataset.poseA_test = poseA_test;
HumanEvaDataset.poseB_test = poseB_test;
HumanEvaDataset.poseC_test = poseC_test;
HumanEvaDataset.poseD_test = poseD_test;
HumanEvaDataset.poseE_test = poseE_test;
HumanEvaDataset.poseF_test = poseF_test;
HumanEvaDataset.poseG_test = poseG_test;
HumanEvaDataset.poseH_test = poseH_test;
HumanEvaDataset.poseI_test = poseI_test;

save('HumanEvaDataset.mat','HumanEvaDataset');

