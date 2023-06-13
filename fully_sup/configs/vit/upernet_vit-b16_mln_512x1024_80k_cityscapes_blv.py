_base_ = [
    '../_base_/models/upernet_vit-b16_ln_mln.py',
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

model = dict(
    pretrained='pretrain/vit_base_patch16_224.pth',
    decode_head=dict(
        num_classes=19,
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
    auxiliary_head=dict(num_classes=19))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
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

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)
