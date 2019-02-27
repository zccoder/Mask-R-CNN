#!/usr/bin/env bash

cd ~/Applications/dev/PycharmProjects/df_vedio_segment/data
ll train_color | awk '{print $9}' | awk -F '_' '{print $4}'| sort | uniq -c

# 19616 5.jpg
# 19606 6.jpg



cd ~/Applications/dev/PycharmProjects/df_vedio_segment/data

#本地测试集数据
cp ~/Documents/ml_data/datafountain/test/56dbd8514bd2b8d1566f8977cfeb0406.jpg test
cp ~/Documents/ml_data/datafountain/test/632c2285b6c98a9973b87ee22bb3269c.jpg test
cp ~/Documents/ml_data/datafountain/test/45b6cc37de77e45411701fcc597c602a.jpg test
cp ~/Documents/ml_data/datafountain/test/cb074e50a78d8ecd31f93bfa0b468669.jpg test


#本地训练数据
cp ~/Documents/ml_data/datafountain/train_color/170908_061502408_Camera_5.jpg train_color/
cp ~/Documents/ml_data/datafountain/train_color/170908_061502547_Camera_5.jpg train_color/
cp ~/Documents/ml_data/datafountain/train_color/170908_061955478_Camera_5.jpg train_color/
cp ~/Documents/ml_data/datafountain/train_color/170908_061955628_Camera_5.jpg train_color/
cp ~/Documents/ml_data/datafountain/train_color/170927_063838664_Camera_5.jpg train_color/
cp ~/Documents/ml_data/datafountain/train_color/170927_063838805_Camera_5.jpg train_color/
cp ~/Documents/ml_data/datafountain/train_color/170927_064342946_Camera_5.jpg train_color/
cp ~/Documents/ml_data/datafountain/train_color/170927_064343082_Camera_5.jpg train_color/


#本地标签数据
cp ~/Documents/ml_data/datafountain/train_label/170908_061502408_Camera_5_instanceIds.png train_label/
cp ~/Documents/ml_data/datafountain/train_label/170908_061502547_Camera_5_instanceIds.png train_label/
cp ~/Documents/ml_data/datafountain/train_label/170908_061955478_Camera_5_instanceIds.png train_label/
cp ~/Documents/ml_data/datafountain/train_label/170908_061955628_Camera_5_instanceIds.png train_label/
cp ~/Documents/ml_data/datafountain/train_label/170927_063838664_Camera_5_instanceIds.png train_label/
cp ~/Documents/ml_data/datafountain/train_label/170927_063838805_Camera_5_instanceIds.png train_label/
cp ~/Documents/ml_data/datafountain/train_label/170927_064342946_Camera_5_instanceIds.png train_label/
cp ~/Documents/ml_data/datafountain/train_label/170927_064343082_Camera_5_instanceIds.png train_label/



39222
3922 val
35900 train

################################################################
cp ~/Documents/ml_data/datafountain/

ls train_color/ | wc -l
39222

cat train_video_list/* | wc -l
42342

ls test/  | wc -l
1917

cat test_video_list_and_name_mapping/list_test/* | wc -l
1907

cat test_video_list_and_name_mapping/list_test_mapping/* | wc -l
1907
