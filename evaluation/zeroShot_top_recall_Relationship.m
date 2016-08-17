% this code is revised based on ILSVRC 2013 (http://www.image-net.org/challenges/LSVRC/2013/)

function  zeroShot = zeroShot_top_recall_Relationship(Nre, tuple_confs_cell, tuple_labels_cell, sub_bboxes_cell, obj_bboxes_cell)

load('gt.mat','gt_tuple_label','gt_obj_bboxes','gt_sub_bboxes');
load('zeroShot.mat','zeroShot');

for ii = 1 : 1000 
    det = find(zeroShot{ii} == 0);
    gt_tuple_label{ii}(det,:) = [];
    gt_obj_bboxes{ii}(det,:) = [];
    gt_sub_bboxes{ii}(det,:) = [];
end
 
%num_imgs = length(gt_tuple_label);
num_imgs = 1000;
for i=1:num_imgs
    [tuple_confs_cell{i}, ind] = sort(tuple_confs_cell{i},'descend');
    if length(ind) >= Nre
        tuple_confs_cell{i} = tuple_confs_cell{i}(1:Nre);
        tuple_labels_cell{i} = tuple_labels_cell{i}(ind(1:Nre),:);
        obj_bboxes_cell{i} = obj_bboxes_cell{i}(ind(1:Nre),:);
        sub_bboxes_cell{i} = sub_bboxes_cell{i}(ind(1:Nre),:);
    else
        tuple_labels_cell{i} = tuple_labels_cell{i}(ind,:);
        obj_bboxes_cell{i} = obj_bboxes_cell{i}(ind,:);
        sub_bboxes_cell{i} = sub_bboxes_cell{i}(ind,:);
    end
end

num_pos_tuple = 0;
for ii = 1 : num_imgs
    num_pos_tuple = num_pos_tuple + size(gt_tuple_label{ii},1);
end
 

tp_cell = cell(1,num_imgs);
fp_cell = cell(1,num_imgs);

gt_thr = 0.5;

% iterate over images
for i=1:num_imgs 
 
    gt_tupLabel = gt_tuple_label{i};
    gt_objBox = gt_obj_bboxes{i};
    gt_subBox = gt_sub_bboxes{i};
     
    num_gt_tuple = size(gt_tupLabel,1);
    gt_detected = zeros(1,num_gt_tuple);
   
    labels = tuple_labels_cell{i};
    boxObj = obj_bboxes_cell{i};
    boxSub = sub_bboxes_cell{i};

    num_obj = size(labels,1);
    tp = zeros(1,num_obj);
    fp = zeros(1,num_obj);
    for j=1:num_obj

        bbO = boxObj(j,:);
        bbS = boxSub(j,:);
        ovmax = -inf;
        kmax = -1;
        
        for k=1:num_gt_tuple
            if norm(labels(j,:) - gt_tupLabel(k,:),2) ~= 0
                continue;
            end
            if gt_detected(k) > 0
                continue;
            end
            
            bbgtO = gt_objBox(k,:);
            bbgtS = gt_subBox(k,:);
            
            biO=[max(bbO(1),bbgtO(1)) ; max(bbO(2),bbgtO(2)) ; min(bbO(3),bbgtO(3)) ; min(bbO(4),bbgtO(4))];
            iwO=biO(3)-biO(1)+1;
            ihO=biO(4)-biO(2)+1;
        
            biS=[max(bbS(1),bbgtS(1)) ; max(bbS(2),bbgtS(2)) ; min(bbS(3),bbgtS(3)) ; min(bbS(4),bbgtS(4))];
            iwS=biS(3)-biS(1)+1;
            ihS=biS(4)-biS(2)+1;
            
            if iwO>0 & ihO>0 &  iwS>0 & ihS>0                  
                % compute overlap as area of intersection / area of union
                uaO=(bbO(3)-bbO(1)+1)*(bbO(4)-bbO(2)+1)+...
                   (bbgtO(3)-bbgtO(1)+1)*(bbgtO(4)-bbgtO(2)+1)-...
                   iwO*ihO;
                ovO =iwO*ihO/uaO;
                
                uaS=(bbS(3)-bbS(1)+1)*(bbS(4)-bbS(2)+1)+...
                   (bbgtS(3)-bbgtS(1)+1)*(bbgtS(4)-bbgtS(2)+1)-...
                   iwS*ihS;
                ovS =iwS*ihS/uaS;
                ov = min([ovO,ovS]);
                
                % makes sure that this object is detected according
                % to its individual threshold
                if ov >= gt_thr && ov > ovmax
                    ovmax=ov;
                    kmax=k;
                end
            end
        end
        
        if kmax > 0
            tp(j) = 1;
            gt_detected(kmax) = 1;
        else
            fp(j) = 1;
        end
    end

    % put back into global vector
    tp_cell{i} = tp;
    fp_cell{i} = fp;


end 

t = tic;
tp_all = [];
fp_all = [];
confs = [];
for ii = 1 : num_imgs
tp_all = [tp_all; tp_cell{ii}(:) ];
fp_all = [fp_all; fp_cell{ii}(:) ];
confs = [confs; tuple_confs_cell{ii}(:)];
end

[confs, ind] = sort(confs,'descend');
tp_all = tp_all(ind);
fp_all = fp_all(ind); 

 
tp = cumsum(tp_all );
fp = cumsum(fp_all );
recall =(tp/num_pos_tuple);
zeroShot = recall(end);

end