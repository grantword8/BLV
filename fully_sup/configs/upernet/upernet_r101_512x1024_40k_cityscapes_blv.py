_base_ = './upernet_r50_512x1024_40k_cityscapes.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101),
             decode_head=dict(
                 loss_decode=dict(
                     _delete_=True,
                     type='BlvLoss',
                     cls_num_list=[16099193283, 4155428633, 8478285320, 925367586, 317481282, 531401266,
                                   67288808, 40463971, 3810296251, 1074861777, 6784666433, 181828023,
                                   15321782, 259269812, 564950248, 184638969, 32522117, 15799033,
                                   2718199],
                     sigma=4,
                 )
             ),
             )
