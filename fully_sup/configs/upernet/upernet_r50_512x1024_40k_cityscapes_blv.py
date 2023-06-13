_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
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