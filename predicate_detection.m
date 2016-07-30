%   This file is for Predicting predicate 

%   Distribution code Version 1.0 -- Copyright 2016, AI lab @ Stanford University.
%   
%   The Code is created based on the method described in the following paper 
%   [1] "Visual Relationship Detection with Language Priors", 
%   Cewu Lu*, Ranjay Krishna*, Michael Bernstein, Li Fei-Fei, European Conference on Computer Vision, 
%   (ECCV 2016), 2016(oral). (* = indicates equal contribution)
%  
%   The code and the algorithm are for non-comercial use only.

%% data loading
addpath('evaluation');
load('data/objectListN.mat'); 
% given a object category index and ouput the name of it.

load('data/obj2vec.mat'); 
% word-to-vector embeding based on https://github.com/danielfrg/word2vec
% input a word and ouput a vector.

load('data/UnionCNNfeaPredicate.mat')
% the CNN score on union of the boundingboxes of the two participating objects in that relationship. 
% we provide our scores (VGG based) here, but you can re-train a new model.

load('data/objectDetRCNN.mat');
% object detection results. The scores are mapped into [0,1]. 
% we provide detected object (RCCN with VGG) here, but you can use a better model (e.g. ResNet).
% three items: 
% detection_labels{k}: object category index in k^{th} testing image.
% detection_bboxes{k}: detected object bounding boxes in k^{th} testing image. 
% detection_confs{k}: confident score vector in k^{th} testing image. 

load('data/Wb.mat');
% W and b in Eq. (2) in [1]

%% We assume we have ground truth object detection
% we will change "predicate" in rlp_labels_ours use our prediction

load('evaluation/gt.mat');
rlp_labels_ours = gt_tuple_label; 
sub_bboxes_ours = gt_sub_bboxes;
obj_bboxes_ours = gt_obj_bboxes;

% gt_tuple_label{j}(k,:) is a tuple that record categroy indexes of <subject, predicate, object>
% gt_sub_bboxes{j}: bounding boxes of subject 
% obj_bboxes_ours{j}: bounding boxes of object 

%% 
testNum = 1000;
fprintf('\n');
fprintf('#######  Predicate computing Begins  ####### \n');
for ii = 1 : testNum
    
    if mod(ii, 100) == 0
        fprintf([num2str(ii), 'th image is tested! \n']);
    end
    
    len = size(gt_tuple_label{ii},1);
    if len ~= 0
        rlp_confs_ours{ii} = zeros(len, 1); 
    else
        rlp_confs_ours{ii} = []; 
    end
  
    for jj = 1 : len
 
        % language modual
        k1 = rlp_labels_ours{ii}(jj,1);
        k2 = rlp_labels_ours{ii}(jj,3);
        vec_org  = [obj2vec(objectListN{k1}),obj2vec(objectListN{k2}),1];
        languageModual =  [W,B]*vec_org';

        % vision modual
        visualModual = max(UnionCNNfeaPredicate{ii}(jj,:),1) ;

        % score vector over relationship
        rlpScore = (languageModual').*visualModual;
         
        [m_score, m_preidcate]  = max(rlpScore); 
        rlp_labels_ours{ii}(jj,2) = m_preidcate;            
        rlp_confs_ours{ii}(jj) = m_score;   
    end
 
end

%% sort by confident score

for ii = 1 : length(rlp_confs_ours)
    [Confs, ind] = sort(rlp_confs_ours{ii}, 'descend');
    rlp_confs_ours{ii} = Confs;
    rlp_labels_ours{ii} = rlp_labels_ours{ii}(ind,:);
    sub_bboxes_ours{ii} = sub_bboxes_ours{ii}(ind,:);
    obj_bboxes_ours{ii} = obj_bboxes_ours{ii}(ind,:);
end


save('results/predicate_det_result.mat', 'rlp_labels_ours', 'rlp_confs_ours', 'sub_bboxes_ours', 'obj_bboxes_ours');

%% computing Predicate Det. accuracy
fprintf('\n');
fprintf('#######  Top recall results  ####### \n');
recall100R = top_recall_Relationship(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
recall50R = top_recall_Relationship(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
fprintf('Predicate Det. R@100: %0.2f \n', 100*recall100R);
fprintf('Predicate Det. R@50: %0.2f \n', 100*recall50R);

fprintf('\n');
fprintf('#######  Zero-shot results  ####### \n');
zeroShot100R = zeroShot_top_recall_Relationship(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
zeroShot50R = zeroShot_top_recall_Relationship(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
fprintf('zero-shot Predicate Det. R@100: %0.2f \n', 100*zeroShot100R);
fprintf('zero-shot Predicate Det. R@50: %0.2f \n', 100*zeroShot50R);



