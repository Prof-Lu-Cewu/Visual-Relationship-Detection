function result_visualization(id, idx, saveFile, rlp_labels_ours, rlp_confs_ours, sub_bboxes_ours, obj_bboxes_ours)

    dataset_test = 'samples'; 
    
    load('data/imagePath.mat')    
    load('data/relationListN.mat')
    load('data/objectListN.mat')
    if ~exist([dataset_test,'/',imagePath{id}])
        disp('please download scene graph dataset form')
        disp('############')
        return;
    end
    
    im = im2double(imread([dataset_test,'/',imagePath{id}]));
    vw = 4; 

    box1 = sub_bboxes_ours{id}(idx,:);
    box2 = obj_bboxes_ours{id}(idx,:);

    strObject1 = char(objectListN(rlp_labels_ours{id}(idx,1)));
    strObject2 = char(objectListN(rlp_labels_ours{id}(idx,3)));
    strRelationship = char(relationListN(rlp_labels_ours{id}(idx,2)));

    mask = zeros(size(im));
    masks = zeros(size(im));
    mask(box1(2):box1(4),box1(1):box1(3),:) = 1;
    masks((box1(2)+vw):(box1(4)-vw),(box1(1)+vw):(box1(3)-vw),:) = 1; 
    mask = (mask - masks);
    mask(:,:,2:3) = -10*mask(:,:,2:3);
    im = min(max(im + mask,0),1); 

    mask = zeros(size(im));
    masks = zeros(size(im));
    mask(box2(2):box2(4),box2(1):box2(3),:) = 1;
    masks((box2(2)+vw):(box2(4)-vw),(box2(1)+vw):(box2(3)-vw),:) = 1; 
    mask = (mask - masks);
    mask(:,:,1:2) = -10*mask(:,:,1:2);
    im = min(max(im + mask,0),1); 

    ob1.cx = round((box1(2) + box1(4))/2);
    ob1.cy = round((box1(1) + box1(3))/2);

    ob2.cx = round((box2(2) + box2(4))/2);
    ob2.cy = round((box2(1) + box2(3))/2);

    rel.cx = round((ob1.cx + ob2.cx)/2);
    rel.cy = round((ob1.cy + ob2.cy)/2);
    
   
    gcf=figure;  imshow(im);hold on
    text(ob1.cy, ob1.cx,strObject1,'color','red','fontsize',20);hold on
    text(ob2.cy, ob2.cx,strObject2,'color','blue','fontsize',20); hold on
    if size(im,1) > size(im,2)
        strRep = [ '<',strObject1 , ', ' ,strRelationship ,', ',  strObject2 , '> score: ', sprintf('%0.1f',rlp_confs_ours{id}(idx))];
        text(1, round(size(im,2)/10), strRep,'color','green','fontsize',18);hold on
    else
        strRep = [ '<',strObject1 , ', ' ,strRelationship ,', ',  strObject2 , '> score: ', sprintf('%0.1f',rlp_confs_ours{id}(idx))];
        text(round(size(im,1)/7), round(size(im,2)/10), strRep,'color','green','fontsize',20);hold on
    end
    saveas(gcf,[saveFile, num2str(id),'_', num2str(idx)],'png');
    close all;
  
end
