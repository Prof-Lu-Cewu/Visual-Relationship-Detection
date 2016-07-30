%   This file is for Predicting <subject, predicate, object> phrase and relationship

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

load('data/UnionCNNfea.mat'); 
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

testNum = 1000;
fprintf('#######  Relationship computing Begins  ####### \n');
for ii = 1 : testNum
    
    if mod(ii, 100) == 0
        fprintf([num2str(ii), 'th image is tested! \n']);
    end
    
    rlp_labels_ours{ii} = [];
    rlp_confs_ours{ii} = []; 
    sub_bboxes_ours{ii} = []; 
    obj_bboxes_ours{ii} = [];

    detL = double(detection_labels{ii});
    detB = double(detection_bboxes{ii}); 
    detC = double(detection_confs{ii});

    uu = 0;
    for k1 = 1 : size(detection_bboxes{ii},1)
        for k2 = 1 : size(detection_bboxes{ii},1)
            if k1 ~= k2 
                uu = uu + 1;                
                
                % language modual
                vec_org  = [obj2vec(objectListN{detL(k1)}),obj2vec(objectListN{detL(k2)}),1];
                languageModual =  [W,B]*vec_org';
                
                % vision modual
                visualModual = detC(k1)*detC(k2)*max(UnionCNNfea{ii}(uu,:),1) ;
                
                % score vector over predicates
                rlpScore = (languageModual').*visualModual;
                
                % selecting best  <subject, predicate, object> tuple
                [m_score, m_preidcate]  = max(rlpScore); 
               
                rlp_labels_ours{ii} = [rlp_labels_ours{ii}; [detL(k1), m_preidcate, detL(k2)]];
                % relationship labels is indexes of <subject, predicate, object>
                
                rlp_confs_ours{ii} = [rlp_confs_ours{ii}; m_score];   
                sub_bboxes_ours{ii} = [sub_bboxes_ours{ii};detB(k1,:) ];  
                obj_bboxes_ours{ii} = [obj_bboxes_ours{ii};detB(k2,:) ];

            end
        end
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

%% 
save('results/relationship_det_result.mat', 'rlp_labels_ours', 'rlp_confs_ours', 'sub_bboxes_ours', 'obj_bboxes_ours');
 
%% computing Phrase Det. and Relationship Det. accuracy

fprintf('\n');
fprintf('#######  Top recall results  ####### \n');
recall100P = top_recall_Phrase(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
recall50P = top_recall_Phrase(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours); 
fprintf('Phrase Det. R@100: %0.2f \n', 100*recall100P);
fprintf('Phrase Det. R@50: %0.2f \n', 100*recall50P);

recall100R = top_recall_Relationship(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
recall50R = top_recall_Relationship(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
fprintf('Relationship Det. R@100: %0.2f \n', 100*recall100R);
fprintf('Relationship Det. R@50: %0.2f \n', 100*recall50R);

fprintf('\n');
fprintf('#######  Zero-shot results  ####### \n');
zeroShot100P = zeroShot_top_recall_Phrase(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
zeroShot50P = zeroShot_top_recall_Phrase(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
fprintf('zero-shot Phrase Det. R@100: %0.2f \n', 100*zeroShot100P);
fprintf('zero-shot Phrase Det. R@50: %0.2f \n', 100*zeroShot50P);

zeroShot100R = zeroShot_top_recall_Relationship(100, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
zeroShot50R = zeroShot_top_recall_Relationship(50, rlp_confs_ours, rlp_labels_ours, sub_bboxes_ours, obj_bboxes_ours);
fprintf('zero-shot Relationship Det. R@100: %0.2f \n', 100*zeroShot100R);
fprintf('zero-shot Relationship Det. R@50: %0.2f \n', 100*zeroShot50R);