_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b0_20220624-7e0fe6dd.pth')),
    decode_head=dict(
        loss_decode=dict(
            _delete_=True,
            type='BlvLoss',
            cls_num_list=[16099193283, 4155428633, 8478285320, 925367586, 317481282, 531401266,
                          67288808, 40463971, 3810296251, 1074861777, 6784666433, 181828023,
                          15321782, 259269812, 564950248, 184638969, 32522117, 15799033,
                          2718199],
            sigma=4,
        ),
    ),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=1, workers_per_gpu=1)
